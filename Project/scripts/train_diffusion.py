import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from trainers.trainer_diffusion import DiffusionTrainer
from dataloaders.weights import WeightDataset
from tqdm import tqdm

def load_data(path):
    files = []
    
    for i in range(1, 4):
        Ri_files = os.listdir(os.path.join(path, f'R{i}'))
        print(f"Found {len(Ri_files)} files at R{i} ")
        for f in tqdm(Ri_files):
            files.append(os.path.join(path, f"R{i}", f))  
            
    return pd.DataFrame({
        'files': files
    }).sample(frac=1.0)
    

def run(args):
    device = args.device
    
    PATH = '/scratch/fk'
    df = load_data(PATH)
    print(df)
    from sklearn.model_selection import train_test_split
    
    df_train, df_val = train_test_split(df, test_size=0.2)
    train_dataset =  WeightDataset(df_train, add_noise=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=7)
    
    val_dataset = WeightDataset(df_val, add_noise=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7)
    print("DataLoaders created")
    
    
    trainer = DiffusionTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=1e-4
    )
    
    trainer.train()
    
    
if __name__ == "__main__":
    pass
    
    
    