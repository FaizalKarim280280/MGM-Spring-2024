import torch
import torch.nn as nn
from icecream import ic
import numpy as np

class DiffusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = self.create_fc_layers(512, 256)
        self.fc2 = self.create_fc_layers(256, 256)
        self.fc3 = self.create_fc_layers(256, 256)
        self.fc4 = self.create_fc_layers(256, 256)
        self.fc5 = self.create_fc_layers(256, 512, last=True)
        
        self.time_fc = self.create_fc_layers(128, 128)
        self.layer_fc = self.create_fc_layers(128, 128)
        
        
    def create_fc_layers(self, in_, out_, last=False):
        return nn.Sequential(
            nn.Linear(in_, out_//2, bias=False),
            nn.BatchNorm1d(out_//2),
            nn.LeakyReLU(0.1),
            nn.Linear(out_//2, out_, bias= not last),
            nn.BatchNorm1d(out_) if not last else nn.Identity(),
            nn.LeakyReLU(0.1) if not last else nn.Identity()
        )
        
    def create_pos_emb_fc(self, in_, out_):
        return nn.Linear(
            nn.Linear(in_, out_),
            nn.Tanh()
        )
        
    def positional_encodings(self, indices, embedding_dim):
        max_length = 200
        # Create positions tensor
        positions = torch.arange(max_length).unsqueeze(1).float()  # Shape: (max_length, 1)

        # Compute div_term
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))  # Shape: (embedding_dim / 2)

        # Compute positional encodings
        pos_encodings = torch.zeros(max_length, embedding_dim, device=indices.device)  # Shape: (max_length, embedding_dim)
        pos_encodings[:, 0::2] = torch.sin(positions * div_term)
        pos_encodings[:, 1::2] = torch.cos(positions * div_term)

        # Gather positional encodings for the input indices
        pos_encodings = pos_encodings[indices, :]  # Shape: (batch_size, embedding_dim)

        return pos_encodings

    def forward(self, x, time, layer):
        # ic(self.positional_encodings(time, embedding_dim=128).device)
        time_emb = self.time_fc(self.positional_encodings(time, embedding_dim=128)) # bs, 128
        layer_emb = self.layer_fc(self.positional_encodings(layer, embedding_dim=128)) # bs, 128
        emb = torch.cat([time_emb, layer_emb], dim=-1) # 256
        
        x = self.fc1(x) # bs, 512
        x = self.fc2(x + emb) # bs, 256
        x = self.fc3(x + emb) # bs, 256
        x = self.fc4(x + emb) # bs, 256
        x = self.fc5(x) # bs, 512
        
        return x
            

class Diffusion:
    def __init__(self,
                 noise_steps,
                 beta_start,
                 beta_end,
                 device):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def add_noise(self, x, t):
        # x is the latent vector: bs, 512
        alpha_t = self.alpha_hat[t]
        mean_t = torch.sqrt(alpha_t)[:, None].to(self.device)
        var_t = torch.sqrt(1 - alpha_t)[:, None].to(self.device)
        
        epsilon = torch.rand_like(x).to(self.device)
        
        # print(mean_t.shape, var_t.shape)
        
        return (mean_t * x) + (var_t * epsilon), epsilon
    
    def remove_noise(self, x, pred, time):
        alpha = self.alpha[time][:, None].to(self.device)
        alpha_hat = self.alpha_hat[time][:, None].to(self.device)
        beta = self.beta[time][:, None].to(self.device)
        
        noise = torch.zeros(len(time), x.shape[1])  
        nonzero_indices = (time != 0).nonzero(as_tuple=True)[0]
        noise[nonzero_indices] = torch.randn(len(nonzero_indices), x.shape[1])
        noise = noise.to(self.device)
        
        a = torch.sqrt(1/alpha)
        b = (1 - alpha)/(torch.sqrt(1 - alpha_hat))

             
        return a * (x - b * pred) + (beta * noise)

def main():
    device = 'cuda'
    diffusion = Diffusion(
        noise_steps=100,
        beta_start=1e-5,
        beta_end=1e-3,
        device=device
    )
    
    t = diffusion.sample_timesteps(32).to(device)
    layer = torch.tensor([(2+1) * 10] * 32).to(device)
    
    ic(layer, layer.dtype)
    ic(t, t.dtype)
    
    x = torch.randn(32, 512).to(device)
    x_noised, noise = diffusion.add_noise(x, t)
    
    model = DiffusionNet().to(device)
    
    pred = model(x_noised, t, layer)
    ic(pred.shape, noise.shape)
    
    
    

if __name__ == "__main__":
    main()