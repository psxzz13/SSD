import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class CoTNetLayer(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
          
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1, stride=1, bias=False),  
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(  
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),  
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1) 
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  
        v = self.value_embed(x).view(bs, c, -1)  

        y = torch.cat([k1, x], dim=1)  
        att = self.attention_embed(y)  
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  
        k2 = F.softmax(att, dim=-1) * v  
        k2 = k2.view(bs, c, h, w)

        return k1 + k2 
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    cot = CoTNetLayer(dim=512, kernel_size=3)
    output = cot(input)
    print(output.shape)