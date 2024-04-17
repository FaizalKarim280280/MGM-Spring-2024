import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

train_dataset = datasets.MNIST(root='/scratch/fk/temp/data', 
                               train=True, 
                               transform=transform,
                               download=True)
val_dataset = datasets.MNIST(root='/scratch/fk/temp/data', 
                              train=False, 
                              transform=transform,
                              download=True)


def main():
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    a, b = next(iter(val_loader))
    torch.save(
        {
            'X': a,
            'y': b
        }, './mnist-val.pth')
    
if __name__ == "__main__":
    main()
    
    