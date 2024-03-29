import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils as vutils

class Trainer:
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 device,
                 lr=1e-4):
        
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.commitment_cost = 0.4
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.mse_loss = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20, 30], gamma=0.5)
        
        
    def get_loss(self, x, x_hat, z, z_q):
        recon_loss = self.mse_loss(x, x_hat)
        commitment_loss = self.mse_loss(z_q.detach(), z)
        quantization_loss = self.mse_loss(z_q, z.detach())
        
        return recon_loss, quantization_loss, commitment_loss
        
    def training_step(self, x):
        out = self.model(x)
        x_hat, z, z_q = out.values()
        recon_loss, quantization_loss, commitment_loss = self.get_loss(x, x_hat, z, z_q)
        loss = recon_loss + quantization_loss + 0.4 * commitment_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        
        return recon_loss, quantization_loss, commitment_loss
    
    def val_step(self, x):
        with torch.no_grad():
            out = self.model(x)
            x_hat, z, z_q = out.values()
            recon_loss, quantization_loss, commitment_loss = self.get_loss(x, x_hat, z, z_q)
            # loss = recon_loss + quantization_loss + self.commitment_cost * commitment_loss
            
            return recon_loss, quantization_loss, commitment_loss
        
    def go_one_epoch(self, loader, step_fxn):
        recon_loss, quant_loss, commit_loss = 0, 0, 0
        for x in tqdm(loader):
            x = x.to(self.device)
            rec_now, quant_now, commit_now = step_fxn(x)
            recon_loss += rec_now
            quant_loss += quant_now
            commit_loss += commit_now
            
        return {
            'recon_loss': recon_loss/len(loader), 
            'quant_loss': quant_loss/len(loader), 
            'commit_loss': commit_loss/len(loader)
            }
        
    def inference(self, epoch, val_loader):
        indices = np.random.choice(32, 25, replace=False)
        img = next(iter(val_loader))[indices].to(self.device)
        
        with torch.no_grad():
            out, _, _ = self.model(img).values()
        
        plt.figure(figsize=(5, 3))
        plt.subplot(1, 2, 1)
        grid_image_input = vutils.make_grid(img, nrow=5, padding=2, normalize=False).permute(1, 2, 0)
        plt.imshow(grid_image_input.cpu().numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        grid_image_out = vutils.make_grid(out, nrow=5, padding=2, normalize=False).permute(1, 2, 0)
        plt.imshow(grid_image_out.cpu().numpy())
        plt.axis('off')
        plt.savefig(f'./temp/outputs/flowers/{epoch}.png')
        plt.close()
    

    def train(self, epochs = 100):
        
        for epoch in range(epochs):
            train_losses = self.go_one_epoch(self.train_loader, self.training_step)
            val_losses = self.go_one_epoch(self.val_loader, self.val_step)
            
            self.inference(epoch, self.val_loader)
            
            print(f"[Epoch:{epoch}] Train:[recon:{train_losses['recon_loss']:.4f} " \
                f"quant:{train_losses['quant_loss']:.4f} " \
                f"commit:{train_losses['commit_loss']:.4f}] " \
                f"Val:[recon:{val_losses['recon_loss']:.4f} " \
                f"quant:{val_losses['quant_loss']:.4f} " \
                f"commit:{val_losses['commit_loss']:.4f}]")

        
        
    
        