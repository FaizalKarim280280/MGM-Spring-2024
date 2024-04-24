import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from tqdm import tqdm

class WeightDataset:
    def __init__(self, 
                 df,
                 add_noise=True):
        
        self.df = df
        self.n_samples = len(self.df)
        self.add_noise = add_noise
        self.noise_scaling = 0.025
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # print(idx)
        weights = torch.load(self.df.iloc[idx, 0], map_location=torch.device('cpu'))
        data = {}
        for i in range(1, 5):
            key = f'conv{i}.0.weight'
            data[key] = weights[key].clone()
                                    
            if self.add_noise:
                data[key] += torch.randn_like(weights[key]) * self.noise_scaling
                
        return data
    
    
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
    
          
def main():
    PATH = '/scratch/fk'
    df = load_data(PATH)
    print(df)
    from sklearn.model_selection import train_test_split
    
    df_train, df_test = train_test_split(df, test_size=0.2)
    train_dataset =  WeightDataset(df_train, add_noise=True)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    
    a = next(iter(train_loader))
    
    print(len(a), a['conv4.0.weight'].shape)
    
    
if __name__ == "__main__":
    main()
    
    
    
