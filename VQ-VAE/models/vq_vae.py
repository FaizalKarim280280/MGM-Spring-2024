import torch
import torch.nn as nn
from .base2 import Encoder, Decoder
from icecream import ic

class VQVAE(nn.Module):
    def __init__(self,
                 num_embeddings:int,
                 embedding_dim:int,
                 device):
        
        super().__init__()
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.encoder = Encoder(in_channels=3, latent_dim=self.embedding_dim).to(self.device)
        self.decoder = Decoder(out_channels=3, latent_dim=self.embedding_dim).to(self.device)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        
    def forward(self, x):
        z = self.encoder(x)
        z_q, indices = self.vector_quantization(z)
        z_q_st = z + (z_q - z).detach()
        out = self.decoder(z_q_st)
        
        return {
            'out': out,
            'z': z,
            'z_q': z_q,
        }
        
        
    def vector_quantization(self, z):
        b, c, h, w = z.shape 
        
        z = z.view(b, -1, self.embedding_dim).unsqueeze(2)
        dist = torch.sum((z - self.embedding.weight[None, None, :, :]) ** 2, dim = -1) ** 0.5
        indices = torch.argmin(dist, dim = -1)
        z_q = self.embedding(indices).permute(0, 2, 1).view(-1, self.embedding_dim, h, w)
        
        return z_q, indices
    
    
def main():
    device = 'cuda'
    model = VQVAE(num_embeddings=256, embedding_dim=64, device=device).to(device)
    
    x = torch.randn(32, 3, 64, 64).to(device)
    out = model(x)
    
    for (k, v) in out.items():
        ic(k, v.shape)
        
        
if __name__ == "__main__":
    main()