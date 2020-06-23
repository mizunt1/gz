import torch.nn as nn
import torch
from layers import ConvBlock
class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    """
    def __init__(self,
                 insize=56, z_dim=10):
        super().__init__()
        self.insize = insize
        self.linear_size = int((insize/8)**2)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ELU(),
            nn.AvgPool2d(2),
            ConvBlock(32),
            nn.Conv2d(32,16),
            nn.ELU()
            ConvBlock(16),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 1)
            nn.ELU()
        )
        self.loc = nn.Linear(self.linear_size, z_dim)
        self.scale = nn.Linear(self.linear_size, z_dim)
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        z_loc = self.loc(x)
        z_scale = self.z_scale(x)
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.outsize = outsize
        self.linear_size = int((outsize/8)**2)
        self.net = nn.Sequential(
            nn.Linear(z_dim, self.linear_size),
            nn.ELU()
            nn.ConvTranspose2d(1, 32, 3, 2, 1),
            nn.ELU(),
            ConvBlock(32),
            nn.ConvTranspose2d(32, 32, 3, 2, 2),
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        loc_img = self.net(z)
        return loc_img
