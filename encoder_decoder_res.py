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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool2d(2),
            ConvBlock(32),
            nn.Conv2d(32,16,kernel_size=3, padding=1),
            nn.ELU(),
            ConvBlock(16),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.AvgPool2d(2),
            nn.ELU()
        )
        self.loc = nn.Linear(self.linear_size, z_dim)
        self.scale = nn.Linear(self.linear_size, z_dim)
        
    def forward(self, x):
        x = x - 0.222
        x = x / 0.156
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        z_loc = self.loc(x)
        z_scale = torch.exp(self.scale(x))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.outsize = outsize
        self.linear_size = int((outsize/8)**2)
        self.linear = nn.Linear(z_dim, self.linear_size)
        self.net = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            ConvBlock(32),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, z):
#        import pdb
#        pdb.set_trace()
        z = self.linear(z)
        z = torch.reshape(z, (-1, 1, int(self.outsize/8), int(self.outsize/8)))
        loc_img = self.net(z)
        return loc_img
