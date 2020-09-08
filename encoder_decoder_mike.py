import torch.nn as nn
import torch
from layers import ConvBlock, UpResBloc, Reshape
class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    """
    def __init__(self,
                 insize=56, z_dim=10):
        super().__init__()
        self.insize = insize
        self.linear_size = int(((insize/16)**2)*16)
        # first conv pair
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # second conv pair
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # third conv
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # fourth conv
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # linear
            Reshape(-1, self.linear_size),
            # linear

        )

        self.loc = nn.Linear(int(self.linear_size), z_dim)
        self.scale = nn.Linear(int(self.linear_size), z_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x - 0.222
        x = x / 0.156
        split = self.net(x)
        split = self.relu(split)
        z_loc = self.loc(split)
        z_scale = torch.exp(self.scale(split))
        return z_loc, z_scale, split


class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.outsize = outsize
        self.linear_size = int((outsize/8)**2)
        self.linear = nn.Linear(z_dim, self.linear_size)
        self.net = nn.Sequential(
            nn.ELU(),
            UpResBloc(1, 32),
            nn.ELU(),
            nn.BatchNorm2d(32),
            ConvBlock(32, bias=False),
            UpResBloc(32, 32),
            nn.ELU(),
            ConvBlock(32, bias=False),
            ConvBlock(32, bias=False),
            UpResBloc(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = self.linear(z)
        z = torch.reshape(z, (-1, 1, int(self.outsize/8), int(self.outsize/8)))
        loc_img = self.net(z)
        return loc_img

if __name__ == "__main__":
    x = torch.zeros([10, 1, 128, 128])
    encoder = Encoder(insize=128)
    decoder = Decoder(outsize=128)
    x = encoder(x)
    x = x[0]
    x = decoder(x)
