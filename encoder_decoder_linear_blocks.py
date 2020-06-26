import torch.nn as nn
import torch
from layers import ConvBlock, LinearBlock
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
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.ELU(),
            nn.AvgPool2d(2),
            ConvBlock(32),
            nn.Conv2d(32,16,kernel_size=5, padding=2),
            nn.ELU(),
            ConvBlock(16),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.AvgPool2d(2),
            nn.ELU()
        )
        self.linear_block = LinearBlock(self.linear_size)
        self.loc = nn.Linear(self.linear_size, z_dim)
        self.scale = nn.Linear(self.linear_size, z_dim)
        
    def forward(self, x):
        x = x - 0.222
        x = x / 0.156
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_block(x)
        z_loc = self.loc(x)
        z_scale = torch.exp(self.scale(x))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.outsize = outsize
        self.linear_size = int((outsize/8)**2)
        self.linear_block = LinearBlock(z_dim)
        self.linear = nn.Linear(z_dim, self.linear_size)

        self.net = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose2d(1, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(32),
            ConvBlock(32),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.ELU(),
            Conv
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = self.linear(z)
        z = torch.reshape(z, (-1, 1, int(self.outsize/8), int(self.outsize/8)))
        loc_img = self.net(z)
        return loc_img

if __name__ == "__main__":
    x = torch.zeros([10, 1, 80, 80])
    encoder = Encoder(insize=80)
    decoder = Decoder(outsize=80)
    x = encoder(x)
    x = x[0]
    x = decoder(x)
    