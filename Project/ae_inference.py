import torch
import torch.nn as nn
import os
from tqdm import tqdm
from models.autoencoder import AutoEncoder
from models.base_cnn import CnnBaseModel
import argparse
from icecream import ic
from trainers.trainer_base_cnn import TrainerBaseCNN

def run_ae(models, data):
    weights = []
    loss = 0
    out_prev = None
    
    for idx, x in enumerate(data.values()):
        y_pred, code = models[idx](x, out_prev)
        weights.append(y_pred)
        out_prev = code.detach().clone()
        loss += nn.MSELoss()(y_pred, x)
        
    print("len weights:", len(weights))
    return weights, loss/4

def load_cnn(args, weights): 
    # print(weights[0].shape)
    cnn_model = CnnBaseModel().to(args.device)
    cnn_model.conv1[0].weight.data = weights[0][10].detach().clone()
    cnn_model.conv2[0].weight.data = weights[1][10].detach().clone()
    cnn_model.conv3[0].weight.data = weights[2][10].detach().clone()
    cnn_model.conv4[0].weight.data = weights[3][10].detach().clone()
    
    return cnn_model

def finetune_cnn(args, cnn_model):
    from torch.utils.data import DataLoader
    from dataloaders.mnist import train_dataset, val_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=7)
    
    trainer_base_cnn = TrainerBaseCNN(
        model=cnn_model,
        train_loader = train_loader,
        val_loader = val_loader,
        device = args.device,
        lr=3e-4,
        save_checkpoints=False,        
    )
    
    trainer_base_cnn.train()


def main():
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    args.device = 'cuda'
    args.checkpoint_path = '/scratch/fk/checkpoint_178_0.00744.pth'
    
    model_l1 = AutoEncoder(num_layers=32, in_c=1, out_c=1, device=args.device)
    model_l2 = AutoEncoder(num_layers=32, in_c=32, out_c=32, device=args.device)
    model_l3 = AutoEncoder(num_layers=64, in_c=32, out_c=32, device=args.device)
    model_l4 = AutoEncoder(num_layers=64, in_c=64, out_c=64, device=args.device)

    ae_weights = torch.load(args.checkpoint_path)

    # models are the each layer AEs
    models = [model_l1, model_l2, model_l3, model_l4]
    
    for i, model in enumerate(models):
        model.load_state_dict(ae_weights[i + 1])
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
    data = torch.load('./val-weights.pth')
    
    generated_weights, loss_weights = run_ae(models, data)
    print(len(generated_weights), loss_weights)
    
    cnn_model = load_cnn(args, generated_weights).to(args.device)
    
    # for name, param in cnn_model.named_parameters():
        # if 'conv' in name:
            # param.requires_grad = False
                
    print(cnn_model)
      
    finetune_cnn(args, cnn_model)
    
    

if __name__ == "__main__":
    main()
    
    
"""
1. Project 1: 
    Base: 4k    
    change in ppt format: 500 
2. Project 2:
    Base: 4k
    Report+ppt in last moment: 1k
    
3. CV assignment: 4k
4. CV assignment copy that got cancelled: 2k
5. GraphNN project: 7k
6. HCI assignment: 4k (deadline: 19th april)

Total: 26,500

"""