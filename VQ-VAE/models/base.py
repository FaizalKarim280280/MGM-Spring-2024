import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from icecream import ic

class Encoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 latent_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        self.conv0 = self.create_conv_blocks(self.in_channels, 16, kernel_size=3, padding=1, stride=1)
        self.conv1 = self.create_conv_blocks(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = self.create_conv_blocks(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = self.create_conv_blocks(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = self.create_conv_blocks(128, self.latent_dim, kernel_size=3, padding=1, stride=1)
        
        print("Encoder created")
        
    def create_conv_blocks(self, in_c, out_c, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),

            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            
            nn.MaxPool2d(stride),
            
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # 3, 64, 64
        x = self.conv0(x) # 16, 64, 64
        x = self.conv1(x) # 16, 32, 32
        x = self.conv2(x) # 64, 16, 16
        x = self.conv3(x) # 128, 8, 8
        x = self.conv4(x) # 128, 8, 8
        return x
    
    
class Decoder(nn.Module):
    def __init__(self,
                 out_channels=1,
                 latent_dim=16):
        
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(self.latent_dim, 128, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
        )
        
        self.conv_z = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim//2, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.latent_dim//2, 1, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1)
        )
    
        self.conv2 = self.create_conv_blocks(256, 128, kernel_size=4, padding=1, stride=2)
        self.conv3 = self.create_conv_blocks(128, 64, kernel_size=4, padding=1, stride=2)
        self.conv4 = self.create_conv_blocks(64, 16, kernel_size=4, padding=1, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, self.out_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )
        
        print("Decoder created")
        
        self.att1 = self.create_attn_block(256)
        self.fc_att1 = self.create_fc_blocks(64, 64)
        self.att2 = self.create_attn_block(128)
        self.fc_att2 = self.create_fc_blocks(64, 256)
        
        self.conv_1x1_1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1)
        )
        
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1)
        )
        
        
        self.sigmoid = nn.Sigmoid()
        
    def create_attn_block(self, out_c):
        return nn.Sequential(
            nn.Conv2d(1, out_c//2, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_c//2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_c//2, out_c, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1)
        )
    
    def create_fc_blocks(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.1)
        )
        
    def create_conv_blocks(self, in_c, out_c, kernel_size, padding, stride, last=False):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(out_c, out_c, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, z):
        # x: self.latent_dim, 8, 8
        x = self.conv0(z) # 256, 8, 8
        z = self.conv_z(z).view(-1, 64) # z: -1, 64
        
        z1 = self.att1(self.fc_att1(z).view(-1, 1, 8, 8))
        x = x + z1 # 256, 8, 8
        x_r1 = self.conv_1x1_1(nn.Upsample(size=(32, 32), mode='nearest')(x)) # 32, 32, 32
         
        x = self.conv2(x) # 128, 16, 16
        
        z2 = self.att2(self.fc_att2(z).view(-1, 1, 16, 16))
        x = x + z2 # 128, 16, 16
        x_r2 = self.conv_1x1_2(nn.Upsample(size=(64, 64), mode='nearest')(x)) # 16, 64, 64
        
        x = self.conv3(x) + 0 * x_r1 # 64, 32, 32
        x = self.conv4(x) + 0 * x_r2 # 16, 64, 64
        
        x = self.conv6(x) # 3, 64, 64
        return x
        

        
        
def main():
    device = 'cuda'
    encoder = Encoder(in_channels=3, latent_dim=128).to(device)
    decoder = Decoder(out_channels=3, latent_dim=128).to(device)
    
    x = torch.randn(32, 3, 64, 64).to(device)
    ic(x.shape)
    
    enc_out = encoder(x)
    ic(enc_out.shape)
    
    dec_out = decoder(enc_out)
    ic(dec_out.shape)    

    

if __name__ == "__main__":
    main()
