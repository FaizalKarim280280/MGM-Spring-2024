import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from trainers.trainer_encoder_decoder import Trainer
from models.autoencoder import AutoEncoder
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
    train_dataset =  WeightDataset(df_train, add_noise=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=7)
    
    val_dataset = WeightDataset(df_val, add_noise=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7)
    print("DataLoaders created")
    
    # a = next(iter(train_loader))
    
    # print(len(a), a['conv4.0.weight'].shape)
    
    model_l1 = AutoEncoder(num_layers=32, in_c=1, out_c=1, device=args.device)
    model_l2 = AutoEncoder(num_layers=32, in_c=32, out_c=32, device=args.device)
    model_l3 = AutoEncoder(num_layers=64, in_c=32, out_c=32, device=args.device)
    model_l4 = AutoEncoder(num_layers=64, in_c=64, out_c=64, device=args.device)
    print("Models created") 
    
    # checkpoint = torch.load('/scratch/fk/ae-checkpoints/checkpoint_0_10.00000.pth')
    # model_l1.load_state_dict(checkpoint[1])
    
    
    trainer = Trainer(
        args=args,
        models=[model_l1, model_l2, model_l3, model_l4],
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=None,
        lr=3e-4,
    )
    print("Trainer object created")
    
    
    print("Starting model training")
    trainer.train()
    
    
def main():
    pass


if __name__ == "__main__":
    main()