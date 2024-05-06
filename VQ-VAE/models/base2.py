import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.LeakyReLU(0.1),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(0.2*x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, stride=2, last=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            nn.MaxPool2d(stride),
            DoubleConv(in_channels, out_channels),
            nn.Dropout(0.1) if not last else nn.Identity()
        )
        # self.emb_layer = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(
        #         emb_dim,
        #         out_channels
        #     ),
        # )

    def forward(self, x):
        x = self.maxpool_conv(x)
        # emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, last=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
            nn.Dropout(0.2) if not last else nn.Identity()
        )

        # self.emb_layer = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(
        #         emb_dim,
        #         out_channels
        #     ),
        # )

    def forward(self, x, skip_x):
        x = self.up(x)
        if skip_x is not None:
            x = 0.2 * self.up(skip_x) + x
        x = self.conv(x)
        # emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x


class Encoder(nn.Module):
    def __init__(self,
                 in_channels:int,
                 latent_dim:int,
                 device='cuda'):
        
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.device = device
        self.inc = DoubleConv(self.in_channels, 64)
        self.down1 = Down(64, 128, stride=2)
        self.down2 = Down(128, 256, stride=2)
        self.down3 = Down(256, self.latent_dim, stride=2, last=True)
        self.sa1 = SelfAttention(128, 32)
        self.sa2 = SelfAttention(256, 16)
        
        
    def forward(self, x):
        # 3, 64, 64
        x = self.inc(x) # 64, 64, 64
        x = self.down1(x) # 128, 32, 32
        x = self.sa1(x)
        x = self.down2(x) # 256, 16, 16
        x = self.sa2(x)
        x = self.down3(x) # latent_dim, 8, 8
        return x
        
class Decoder(nn.Module):
    def __init__(self,
                 out_channels:int,
                 latent_dim:int,
                 device='cuda'):
        
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.device = device
        self.up1 = Up(self.latent_dim, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.sa1 = SelfAttention(256, 16)
        self.sa2 = SelfAttention(128, 32)
        self.sa3 = SelfAttention(64, 64)
        self.outc = nn.Sequential(
            nn.Conv2d(64, self.out_channels, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # latent_dim, 8, 8
        x_res = self.up1(x, skip_x=None) # 256, 16, 16
        x = self.sa1(x_res)
        x_res = self.up2(x, skip_x=x_res) # 128, 32, 32
        x = self.sa2(x_res)
        x_res = self.up3(x, skip_x=x_res) # 64, 64, 64
        x = self.outc(x_res) # 3, 64, 64
        return x
    
    
def main():
    device = 'cuda'
    encoder = Encoder(in_channels=3, latent_dim=32).to(device)
    decoder = Decoder(out_channels=3, latent_dim=32).to(device)
    x = torch.randn(4, 3, 64, 64, device=device)
    enc_out = encoder(x)
    ic(enc_out.shape)
    dec_out = decoder(enc_out)
    ic(dec_out.shape)


if __name__ == "__main__":
    main()