import torch
from torch.utils.data import DataLoader
from models.base_cnn import CnnBaseModel
from trainers.trainer_base_cnn import TrainerBaseCNN
from dataloaders.mnist import train_dataset, val_dataset
import numpy as np
import random
import argparse 

def run(args):
    seed = torch.randint(0, 10000000, size=(1, )).item()
    torch.manual_seed(seed)
    seed = torch.randint(0, 10000000, size=(1, )).item()
    np.random.seed(seed)
    seed = torch.randint(0, 10000000, size=(1, )).item()
    random.seed(seed)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=7,
                              drop_last=True)
    
    val_loader = DataLoader(val_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=7,
                              drop_last=True)
        
    model = CnnBaseModel()
    trainer = TrainerBaseCNN(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=1e-5,
        exp_name=args.exp_name,
        run_index=args.run_index
    )
    
    trainer.train(epochs=10)
    
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    args.batch_size = 128
    args.device = 'cuda'
    args.run_index = 0
    
    run(args)
    
if __name__ == "__main__":
    main()