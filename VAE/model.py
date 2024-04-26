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
        self.conv1 = self.create_conv_blocks(16, 16, kernel_size=3, padding=1, stride=2)
        self.conv2 = self.create_conv_blocks(16, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = self.create_conv_blocks(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = self.create_conv_blocks(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv5 = self.create_conv_blocks(256, 512, kernel_size=3, padding=1, stride=2)
        
        self.conv1x1 = nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1)
        
        self.fc1 = self.create_fc_blocks(128, self.latent_dim, mean=True)
        self.fc2 = self.create_fc_blocks(128, self.latent_dim, mean=False)
        
        self.conv0 = self.create_conv_blocks(self.in_channels, 16, kernel_size=5, padding=2, stride=1)
        
        print("Encoder created")
        
    def create_fc_blocks(self, in_dim, out_dim, mean=True):
        return nn.Sequential(
            # nn.Linear(in_dim, in_dim),
            # nn.LeakyReLU(0.1),
            # nn.Dropout(0.1),
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.1) if mean else nn.ReLU()
        )
        
    def create_conv_blocks(self, in_c, out_c, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),

            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            
            nn.MaxPool2d(stride),
            
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1, bias=False),
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
        x = self.conv4(x) # 256, 4, 4
        x = self.conv5(x) # 512, 2, 2
        x = nn.AvgPool2d(2)(x) # 512, 1, 1
        x = self.conv1x1(x) # 128, 1, 1
        x = x.view(-1, 128)
        
        mean = self.fc1(x)
        logvar = self.fc2(x)
        
        return {
            'mean': mean,
            'logvar': logvar
            }
    
    
class Decoder(nn.Module):
    def __init__(self,
                 out_channels=1,
                 latent_dim=16):
        
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        self.fc1 = self.create_fc_blocks(self.latent_dim, 512)
        
        # self.conv0 = self.create_conv_blocks(128, 128 * 2, kernel_size=4, padding=1, stride=2)
        self.conv1 = self.create_conv_blocks(128, 128, kernel_size=4, padding=1, stride=2)
        self.conv2 = self.create_conv_blocks(128, 64, kernel_size=4, padding=1, stride=2)
        self.conv3 = self.create_conv_blocks(64, 32, kernel_size=4, padding=1, stride=2)
        self.conv4 = self.create_conv_blocks(32, 16, kernel_size=4, padding=1, stride=2)
        self.conv5 = self.create_conv_blocks(16, self.out_channels, kernel_size=4, padding=1, stride=2, last=False)
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, self.out_channels, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid()
        )
        
        print("Decoder created")
        
        
        self.att1 = self.create_attn_block(128)
        self.fc_att1 = self.create_fc_blocks(self.latent_dim, 64)
        self.att2 = self.create_attn_block(64)
        self.fc_att2 = self.create_fc_blocks(self.latent_dim, 256)
        
        self.sigmoid = nn.Sigmoid()
        
    def create_attn_block(self, out_c):
        return nn.Sequential(
            nn.Conv2d(1, int(out_c/2), kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(int(out_c/2), out_c, kernel_size=3, padding=1, stride=1),
            # nn.Tanh(),
            
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
            nn.LeakyReLU() if not last else nn.Identity(),
            nn.Dropout(0.1) if not last else nn.Identity()
        )
        
    def forward(self, z, skips=None):
        # -1, latent_dim
        x = self.fc1(z) # -1, 512
        x = x.view(-1, 128, 2, 2) 

        x = nn.Upsample(size=(4, 4), mode='nearest')(x) # 256, 4, 4
        
        x = self.conv1(x) # 128, 8, 8 
        z1 = self.att1(self.fc_att1(z).view(-1, 1, 8, 8))
        # z1 = self.sigmoid(z1)
        x = x + z1
        
        x = self.conv2(x) # 64, 16, 16
        z2 = self.att2(self.fc_att2(z).view(-1, 1, 16, 16))
        # z2 = self.sigmoid(z2)
        # ic(x.shape, z2.shape)
        x = x + z2
        
        
        x = self.conv3(x) # 32, 32, 32
        x = self.conv4(x) # 16, 64, 64
        # x = self.conv5(x) # 3, 128, 128
        
        x = self.conv6(x)
        return x
        


class Net(nn.Module):
    def __init__(self,
                device):
        
        super().__init__()
        self.device = device
        self.encoder = Encoder(in_channels=3, latent_dim=128).to(self.device)
        self.decoder = Decoder(out_channels=3, latent_dim=128).to(self.device)
            
    def reparam(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mean + std * eps
        
    def forward(self, x):
        encoder_out = self.encoder(x)
        mean, logvar = encoder_out.values()
        z = self.reparam(mean=mean, logvar=logvar)
        z = z.to(self.device)
        out = self.decoder(z)
    
        return out, mean, logvar
        
        
def main():
    device = 'cuda'
    # encoder = Encoder().to(device)
    # decoder = Decoder().to(device)
    
    x = torch.randn(32, 3, 128, 128).to(device)
    
    net = Net(device=device)
    ic(net(x))
    
    # enc_out = encoder(x)
    # mean, logvar, skips = enc_out.values()
    # std = torch.exp(0.5 * logvar)
    
    # ic(mean.shape, logvar.shape)

    # z = mean + std * torch.rand_like(std)
    # ic(z.shape)
    
    # out = decoder(z, skips)
    # ic(out.shape)
    

if __name__ == "__main__":
    main()
