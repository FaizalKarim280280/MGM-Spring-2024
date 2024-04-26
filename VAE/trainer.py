import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.utils as vutils

class Trainer:
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 device, 
                 lr = 3e-4,
                 logger=None):
        
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.lr = lr
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.mse_loss = nn.MSELoss()
    
    def get_loss(self, y_pred, x, mean, logvar):
        # recon_loss = nn.functional.binary_cross_entropy(y_pred, x, reduction='sum')
        recon_loss = nn.functional.mse_loss(y_pred, x, reduction='sum')
        # recon_loss = nn.functional.l1_loss(y_pred, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # recon_loss = nn.functional.mse_loss(y_pred, x)
        # kl_div = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = recon_loss + kl_div
        # loss = 1 * recon_loss + 1* kl_div
        return loss
    
    def training_step(self, x):
        y_pred, mean, logvar = self.model(x)
        loss = self.get_loss(y_pred=y_pred, x=x, mean=mean, logvar=logvar)
        
        self.optimizer.zero_grad()
        loss.backward() 
        self.optimizer.step()
        return loss
    
    def val_step(self, x):
        with torch.no_grad():
            y_pred, mean, logvar = self.model(x)
            loss = self.get_loss(y_pred=y_pred, x=x, mean=mean, logvar=logvar)
        return loss
    
    def go_one_epoch(self, loader, step_fxn):
        loss = 0
        for x in tqdm(loader):
            x = x.to(self.device)
            loss += step_fxn(x)
        
        return loss/len(loader)

    def inference(self, epoch, val_loader):
        indices = np.random.choice(32, 25, replace=False)
        img = next(iter(val_loader))[indices].to(self.device)
        
        with torch.no_grad():
            out, _, _ = self.model(img)
            
        plt.figure(figsize=(5, 3))
        plt.subplot(1, 2, 1)
        grid_image_input = vutils.make_grid(img, nrow=5, padding=2, normalize=False).permute(1, 2, 0)
        plt.imshow(grid_image_input.cpu().numpy())
        plt.axis('off')
        plt.subplot(1, 2, 2)
        grid_image_out = vutils.make_grid(out, nrow=5, padding=2, normalize=False).permute(1, 2, 0)
        plt.imshow(grid_image_out.cpu().numpy())
        plt.axis('off')
        plt.savefig(f'./temp/outputs/anime/{epoch}.png')
        
        # plt.show()
        
        
            
    def train(self, epochs=100):
        
        for epoch in range(epochs):
            
            train_loss = self.go_one_epoch(self.train_loader, self.training_step)
            val_loss = self.go_one_epoch(self.val_loader, self.val_step)
            self.inference(epoch=epoch, val_loader=self.val_loader)
            
            print(f"[Epoch: {epoch}] Train:[loss:{train_loss:.5f}] Val:[loss:{val_loss:.5f}]")    

            torch.save(self.model.state_dict(), f'/scratch/fk/checkpoints/vae_shapes_{epoch}.pth')
        