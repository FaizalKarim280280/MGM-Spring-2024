import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, num_layers, out_c):
        super().__init__()
        self.num_layers = num_layers
        self.out_c = out_c
        self.convs = [self.create_conv_layer(in_c=1, out_c=self.out_c, kernel_size=1) for _ in range(self.num_layers)]
        self.convs = nn.ModuleList(self.convs)
        
        self.fc = nn.Sequential(
            nn.Linear(256, self.num_layers * 3 * 3),
            nn.LeakyReLU(0.1),
            nn.Linear(self.num_layers * 3 * 3, self.num_layers * 3 * 3),
        )
        self.drop = nn.Dropout(0.2)
            
    def create_conv_layer(self, in_c, out_c, kernel_size=3):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(out_c, out_c, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        # bs, 256
        bs = x.size(0)
        x = self.fc(x) # bs, num_layers * 3 * 3
        x = x.view(bs, self.num_layers, 1, 3, 3)
        x = self.drop(x)
        
        xs = []
        for i in range(self.num_layers):
            xs.append(self.convs[i](x[:, i, :]))
        
        xs = torch.stack(xs, dim=1)
        return xs
    
class Encoder(nn.Module):
    def __init__(self, num_layers, in_c):
        super().__init__()
        self.num_layers = num_layers
        self.in_c = in_c
        self.convs = [self.create_conv_layer(in_c=self.in_c, out_c=1, kernel_size=1) for _ in range(self.num_layers)]
        
        self.convs = nn.ModuleList(self.convs)
        self.fc = nn.Sequential(
            nn.Linear(self.num_layers * 3 * 3, self.num_layers * 3 * 3),
            nn.LeakyReLU(0.1),
            nn.Linear(self.num_layers * 3 * 3, 256)
        )
        self.drop = nn.Dropout(0.2)
            
    def create_conv_layer(self, in_c, out_c, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # bs, num_layers, in_c, 3, 3
        xs = []
        bs = x.size(0)
        for i in range(self.num_layers):
            xs.append(self.convs[i](x[:, i, :]))
            
        xs = torch.stack(xs, dim=1)
        # print(xs.shape)
        xs = xs.view(bs, self.num_layers * 3 * 3) # bs, num_layers * 3 * 3
        # xs = self.drop(xs)
        # print(xs.shape)
        xs = self.fc(xs) # bs, 256
        return xs
    
    
class AutoEncoder(nn.Module):
    def __init__(self, 
                 num_layers,
                 out_c,
                 in_c,
                 device):
        
        super().__init__()
        self.num_layers = num_layers
        self.out_c = out_c
        self.in_c = in_c
        self.device = device
        
        self.encoder = Encoder(num_layers=self.num_layers, in_c=self.in_c)
        self.decoder = Decoder(num_layers=self.num_layers, out_c=self.out_c)
        
        self.alpha = 0.5
        
    def forward(self, x, out_prev):
        # x = x + self.alpha * out_prev
        
        code = self.encoder(x)
        if out_prev is None:
            code = code + torch.zeros_like(code)
        else:
            code = code + out_prev * self.alpha
        out = self.decoder(code)
        return out, code
    
    
def main():
    model = AutoEncoder(num_layers=32, in_c=1, out_c=1, device='cuda')
    x = torch.randn(4, 32, 1, 3, 3)
    
    out, code = model(x, torch.zeros_like(x))
    print(out.shape, code.shape)
    
    
if __name__ == "__main__":
    main()