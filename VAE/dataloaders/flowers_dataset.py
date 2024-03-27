import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

class FlowersDataset(Dataset):
    def __init__(self, 
                 df, 
                 type):
        
        super().__init__()
        self.df = df
        self.n_samples = len(self.df)
        self.type = type
        self.transforms = train_transform if self.type else val_transform
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        
        img = plt.imread(self.df.iloc[idx, 0])[:, :, :3]        
        img = cv2.resize(img, (128, 128))
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

def main():
    print(torch.__version__)

if __name__ == "__main__":
    main()