import torch
import torch.nn as nn
from .base import Encoder, Decoder
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
        z_org_shape = z.shape
        # Flatten z to (batch_size, channel_size, num_pixels)
        z = z.permute(0, 2, 3, 1)  # (batch_size, height, width, channel)
        flatten_z = z.contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances between each encoded vector and embeddings
        distances = torch.sum(flatten_z**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flatten_z, self.embedding.weight.t())
        
        # Find nearest embeddings (vector quantization)
        _, indices = torch.min(distances, dim=1)
        z_q = self.embedding(indices).view(z_org_shape)
        
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