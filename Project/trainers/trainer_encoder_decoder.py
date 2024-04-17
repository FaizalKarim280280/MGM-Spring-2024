import torch
import torch.nn as nn
import os
from tqdm import tqdm
from models.base_cnn import CnnBaseModel

class Trainer:
    def __init__(self, 
                 args,
                 models,
                 train_loader,
                 val_loader,
                 device,
                 lr=1e-4,
                 logger=None):
        
        self.args = args
        self.device = device
        self.models = [model.to(self.device) for model in models]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.logger = None
        
        self.loss_fxn = nn.MSELoss()
        self.optimizers = [torch.optim.Adam(self.models[i].parameters(), lr=lr) 
                           for i in range(len(self.models))]
        
        self.cnn_model = CnnBaseModel().to(self.device)
        self.mnist_val = torch.load('./mnist-val.pth')
        self.fc_weight = torch.load('./fc-weight.pth').to(self.device)
        self.fc_bias = torch.load('./fc-bias.pth').to(self.device)
    
        self.mnist_val['X'] = self.mnist_val['X'].to(self.device)
        self.mnist_val['y'] = self.mnist_val['y'].to(self.device)
        
    def training_step(self, x, out_prev, idx):
        y_pred, code = self.models[idx](x, out_prev)
        loss = self.loss_fxn(y_pred, x)
        
        self.optimizers[idx].zero_grad()
        loss.backward()
        self.optimizers[idx].step()
        
        return loss, code
        
    def val_step(self, x, out_prev, idx):
        with torch.no_grad():
            y_pred, code = self.models[idx](x, out_prev)
            loss = self.loss_fxn(y_pred, x)
            
        return loss, code
    
    def go_one_epoch(self, loader, step_fxn):
        loss = 0
        
        for data in tqdm(loader):
            out_prev = None
            loss_temp = 0
            
            for idx, x in enumerate(data.values()):
                x = x.to(self.device)
                l, code = step_fxn(x, out_prev, idx)
                loss_temp += l
                out_prev = code.detach().clone()   
                
            loss += loss_temp/4
            
        return loss/len(loader) 
    
    def inference(self, loader):
        cls_acc = 0
        recon_loss = 0
        
        for data in tqdm(loader, desc='Inference'):
            out_prev = None
            weights = []
            loss = 0
            for idx, x in enumerate(data.values()):
                x = x.to(self.device)
                with torch.no_grad():
                    y_pred, code = self.models[idx](x, out_prev)
                    out_prev = code.detach().clone()
                    weights.append(y_pred)
                    loss += nn.MSELoss()(y_pred, x)
                    # print(y_pred.shape)
                    
            recon_loss += loss/4
            
            batch_cls_acc = 0
            
            for batch in range(len(weights[0])):
                self.cnn_model.conv1[0].weight.data = weights[0][batch].detach().clone()
                self.cnn_model.conv2[0].weight.data = weights[1][batch].detach().clone()
                self.cnn_model.conv3[0].weight.data = weights[2][batch].detach().clone()
                self.cnn_model.conv4[0].weight.data = weights[3][batch].detach().clone()
                
                self.cnn_model.fc.weight.data = self.fc_weight
                self.cnn_model.fc.bias.data = self.fc_bias
                
                # for name, param in self.cnn_model.named_parameters():
                #     print(name, param.shape, param.device)
                                
                with torch.no_grad():
                    y_pred = self.cnn_model(self.mnist_val['X'])
                    batch_cls_acc += (torch.argmax(y_pred, dim=-1) == self.mnist_val['y']).sum()/len(self.mnist_val['X'])
                    
            cls_acc += batch_cls_acc/len(weights[0])
                                
        return cls_acc/len(loader), recon_loss/len(loader)
        
    
    def train(self):
        best_val_loss = 1e5
        
        for epoch in range(self.args.epochs):
            train_loss = self.go_one_epoch(self.train_loader, self.training_step)
            val_loss = self.go_one_epoch(self.val_loader, self.val_step)
            
            # cnn_acc, recon_loss = self.inference(self.val_loader)
            
            print(f"[Epoch:{epoch}] "\
                f"Train:[loss:{train_loss:.5f}] "
                f"Val:[loss:{val_loss:.5f}] ")
                # f"CNN:[acc:{cnn_acc:.5f} recon_loss:{recon_loss:.5f}] ")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                filename = os.path.join(self.args.checkpoint_path,
                                        f"checkpoint_{epoch}_{val_loss:.5f}.pth")
                torch.save({
                    k : v.state_dict() for (k, v) in zip(range(1,5), self.models)
                }, filename)
                
                print(f"Saved checkpoint at {filename}")
            
                
        
def main():
    pass


if __name__ == "__main__":
    main()    
    