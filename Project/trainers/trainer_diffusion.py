import torch
import torch.nn as nn
# import sys
# sys.path.append('../denoising_diffusion_pytorch')
# from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from models.diffusion import Diffusion, DiffusionNet

from models.autoencoder import AutoEncoder
from tqdm import tqdm
from icecream import ic

class DiffusionTrainer:
    def __init__(self,
                 train_loader,
                 val_loader,
                 device,
                 lr=1e-4):
        
        self.device = device
        # self.model = Unet1D(
        #     dim=512,
        #     dim_mults=(1,),
        #     channels=1).to(self.device)
        
        # self.diffusion = GaussianDiffusion1D(
        #     self.model,
        #     seq_length = 512,
        #     timesteps = 100,
        #     objective = 'pred_noise').to(self.device)
        
        self.model = DiffusionNet().to(self.device)
        self.diffusion = Diffusion(
            noise_steps=200,
            beta_start=1e-5,
            beta_end=1e-3,
            device=self.device
        )
        
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fxn = nn.L1Loss()
        
        self.autoencoder = [
            AutoEncoder(num_layers=32, in_c=1, out_c=1, device=self.device), 
            AutoEncoder(num_layers=32, in_c=32, out_c=32, device=self.device),
            AutoEncoder(num_layers=64, in_c=32, out_c=32, device=self.device),
            AutoEncoder(num_layers=64, in_c=64, out_c=64, device=self.device)
        ]
        
        ae_weights = torch.load('/scratch/fk/checkpoint_178_0.00744.pth')
        for i, model in enumerate(self.autoencoder):
            model.load_state_dict(ae_weights[i + 1])
            model = model.to(self.device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False    
                
        ic("Autoencoders loaded")
                
        self.alpha = 0.3
        print("Parameters in model: ", self.get_num_parameters(self.model))
        
        # for name, param in self.model.named_parameters():
        #     print(name, param.shape)
        
    def get_num_parameters(self, model):
        count = 0
        for param in model.parameters():
            count += param.numel()
        return f"{count:,}"
        
    def training_step(self, x, layer, code_prev):
        if code_prev is None:
            x = x + self.alpha * torch.zeros_like(x)
        else:
            x = x + self.alpha * code_prev
        
        time = self.diffusion.sample_timesteps(x.size(0)).to(self.device)
        x_t, noise = self.diffusion.add_noise(x, time)
        pred_noise = self.model(x_t, time, layer)
        loss = self.loss_fxn(pred_noise, noise)
        
        # loss = self.diffusion(x.unsqueeze(1), layer)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def val_step(self, x, layer, code_prev):
        if code_prev is None:
            x = x + self.alpha * torch.zeros_like(x)
        else:
            x = x + self.alpha * code_prev
            
        with torch.no_grad():
            # loss = self.diffusion(x.unsqueeze(1), layer)
            time = self.diffusion.sample_timesteps(x.size(0)).to(self.device)
            x_t, noise = self.diffusion.add_noise(x, time)
            pred_noise = self.model(x_t, time, layer)
            loss = self.loss_fxn(pred_noise, noise)
            
        return loss.item()
        
    def go_one_epoch(self, loader, step_fxn):
        loss = 0
        
        for data in tqdm(loader):
            code_prev = None
            loss_temp = 0
            
            for idx, x in enumerate(data.values()):
                x = x.to(self.device)
                layer = torch.tensor([(idx+1) * 10] * x.size(0), device=self.device).long()
                with torch.no_grad():
                    code = self.autoencoder[idx].encoder(x)
                    
                loss_temp += step_fxn(code, layer, code_prev)
                code_prev = code.detach().clone()
            loss += loss_temp/4
            
        return loss/len(loader)
    
    
    def train(self, epochs=50):
        for epoch in range(epochs):
            train_loss = self.go_one_epoch(self.train_loader, self.training_step)
            val_loss = self.go_one_epoch(self.val_loader, self.val_step)
            
            print(f"[Epochs:{epoch+1}]" \
                    f"Train:[loss:{train_loss:.5f}]" \
                    f"Val:[loss:{val_loss:.5f}]")


