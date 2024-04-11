import os
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from trainer import Trainer
from model import Net
from dataloaders.flowers_dataset import FlowersDataset
import torch
from torch.utils.data import DataLoader

SEED = 28
torch.manual_seed(SEED)

def get_data_to_df(path):
    df = {
        'filename': [],
        'label': []
    }
    
    for folder in os.listdir(path):
        for img in os.listdir(os.path.join(path, folder)):
            img_path = os.path.join(path, folder, img)
            # img = plt.imread(img_path)
            # if len(img.shape) != 3:
            #     print(img_path, end = ' ')
            df['filename'].append(img_path)
            df['label'].append(folder)
            
    df = pd.DataFrame(df).sample(frac=1.0, random_state=SEED)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"Train: {len(df_train)}")
    print(f"Val: {len(df_val)}")
    
    return df_train, df_val

def get_number_of_params(model):
    counter = 0
    for p in model.parameters():
        counter += torch.prod(torch.tensor(p.shape)).cpu().numpy()
    return counter
            
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_root = '/scratch/fk/cropped'
    batch_size = 32
    
    df_train, df_val = get_data_to_df(data_root)
    
    train_dataset = FlowersDataset(df=df_train, type='train')
    val_dataset = FlowersDataset(df=df_val, type='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    
    # a = next(iter(train_loader))
    # print(a.shape)   
    # plt.imsave('./temp/1.png', a[0].cpu().permute(1, 2, 0).numpy())
    
    model = Net(device=device)
    
    print("Number of parameters in model:", get_number_of_params(model))
    
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      device=device)
    
    trainer.train(epochs=100)


if __name__ == "__main__":
    main()