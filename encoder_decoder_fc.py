import torch.nn as nn
import torch
from layers import ConvBlock, Flatten, LinearBlock
class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    """
    def __init__(self,
                 insize=56, z_dim=10):
        super().__init__()
        self.insize = insize
        self.flatten_size = int((self.insize)**2)
        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(self.flatten_size, int(self.flatten_size/2)),
            LinearBlock(int(self.flatten_size/2)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/2), int(self.flatten_size/4)),
            LinearBlock(int(self.flatten_size/4)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/4), int(self.flatten_size/8)),
            nn.ELU(),
        )
        self.loc = nn.Linear(int(self.flatten_size/8), z_dim)
        self.scale = nn.Linear(int(self.flatten_size/8), z_dim)
        
    def forward(self, x):
        x = x - 0.222
        x = x / 0.156

        x = self.net(x)
        z_loc = self.loc(x)
        z_scale = torch.exp(self.scale(x))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.outsize = outsize
        self.flatten_size = int((self.outsize)**2)
        self.net = nn.Sequential(
            nn.Linear(z_dim, int(self.flatten_size/8)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/8), int(self.flatten_size/4)),
            LinearBlock(int(self.flatten_size/4)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/4), int(self.flatten_size/2)),
            LinearBlock(int(self.flatten_size/2)),
            nn.ELU(),
            nn.Linear(int(self.flatten_size/2), self.flatten_size),
            nn.Sigmoid(),
        )
        
    def forward(self, z):
        z = self.net(z)
        z = z.reshape(z.shape[0], self.outsize, self.outsize)
        return z

