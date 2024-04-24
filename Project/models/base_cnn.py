import torch
import torch.nn as nn

import sys
sys.path.append('utils')

from utils.utils import get_num_model_params

class CnnBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.create_conv_blocks(1, 32)
        self.conv2 = self.create_conv_blocks(32, 32)
        self.conv3 = self.create_conv_blocks(32, 64, stride=1)
        self.conv4 = self.create_conv_blocks(64, 64)
        
        
        self.avg_pool = nn.AvgPool2d(3)
        self.fc = nn.Linear(64, 10)
        print("CnnBaseModel created")
        
    def create_conv_blocks(self, in_c, out_c, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            # nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=2, bias=False),
            # nn.ReLU()
        )
        
    def forward(self, x):
        # x: bs, 1, 28, 28
        x = self.conv1(x) # bs, 16, 14, 14
        x = self.conv2(x) # bs, 32, 7, 7
        x = self.conv3(x) # bs, 64, 7, 7
        x = self.conv4(x) # bs, 64, 3, 3
        x = self.avg_pool(x) # bs, 64, 1, 1
        x = x.view(-1, x.size(1)) # bs, 64
        x = self.fc(x)
        return x
    
    
def main():
    device = 'cpu'
    model = CnnBaseModel()
    x = torch.randn(4, 1, 28, 28)
    print("Output shape:", model(x).shape)
    print("Number of parameters:", get_num_model_params(model))
    
if __name__ == "__main__":
    main()