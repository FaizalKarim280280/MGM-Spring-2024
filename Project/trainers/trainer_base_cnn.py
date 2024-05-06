import torch
import torch.nn as nn
from tqdm import tqdm
import os

class TrainerBaseCNN:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 device,
                 lr=1e-4,
                 run_index=None,
                 exp_name=None,
                 save_checkpoints=True):
        
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        
        self.loss_fxn = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.conv1.parameters(), 'lr': 1e-6},
            {'params': self.model.conv2.parameters(), 'lr': 1e-6},
            {'params': self.model.conv3.parameters(), 'lr': 1e-6},
            {'params': self.model.conv4.parameters(), 'lr': 1e-6},
            {'params': self.model.fc.parameters(), 'lr': 1e-3},            
        ])
        
        self.accuracy_fxn = None
        self.run_index = run_index
        self.exp_name = exp_name
        self.save_dir = f'/scratch/fk/checkpoints'
        self.save_checkpoints = save_checkpoints
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def training_step(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_fxn(y_pred, y)
        acc = (torch.argmax(y_pred, axis=-1) == y).sum()/len(y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, acc
    
    def val_step(self, x, y):
        with torch.no_grad():
            y_pred = self.model(x)
            loss = self.loss_fxn(y_pred, y)
            acc = (torch.argmax(y_pred, axis=-1) == y).sum()/len(y)            
        return loss, acc
    
    def go_one_epoch(self, loader, step_fxn):
        loss, acc = 0, 0
        for data in tqdm(loader):            
            x, y = data[0].to(self.device), data[1].to(self.device)
            loss_epoch, acc_epoch = step_fxn(x, y)
            loss += loss_epoch.item()
            acc += acc_epoch.item()
            
        return {
            'loss': round(loss/len(loader), 4),
            'metrics': {
                'acc': round(acc/len(loader), 4)
            }
        }
        
    def train(self, epochs=20):
        for epoch in range(epochs):
            train_outputs = self.go_one_epoch(self.train_loader, self.training_step)
            val_outputs = self.go_one_epoch(self.val_loader, self.val_step)
            
            print(f"[Epoch:{epoch}] [Train:{train_outputs}] [Val:{val_outputs}]")
            
            if self.save_checkpoints:    
                if val_outputs['metrics']['acc'] > 0.95:
                    filename = os.path.join(f"{self.save_dir}", f"{self.exp_name}_{self.run_index}_{epoch}.pt")
                    torch.save(self.model.state_dict(), filename)
                    print("Checkpoint saved:", filename)