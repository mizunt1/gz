import torch.nn as nn
import torch
class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    """
    def __init__(self,
                 insize=56, z_dim=10):
        super().__init__()
        self.insize = insize
        self.linear_size = int((insize/8)**2)
        self.layer1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.avgpool = nn.AvgPool2d(2)
        self.layer2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.layer3 = nn.Conv2d(32, 1, 3, 1, 1)
        self.layer41 = nn.Linear(self.linear_size, z_dim)
        self.layer42 = nn.Linear(self.linear_size, z_dim)
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = x - 0.222
        x = x / 0.156
        x = self.layer1(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.layer2(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.layer3(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        z_loc = self.layer41(x)
        z_scale = torch.exp(self.layer42(x))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim=10, outsize=56):
        super().__init__()
        self.outsize = outsize
        self.linear_size = int((outsize/8)**2)
        self.layer1 = nn.Linear(z_dim, self.linear_size)
        self.layer2 = nn.ConvTranspose2d(1, 32, 3, 2, 1)
        self.layer3 = nn.ConvTranspose2d(32, 32, 3, 2)
        self.layer4 = nn.ConvTranspose2d(32, 1, 4, 2)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        
    def forward(self, z):
        z = self.layer1(z)
        z = torch.reshape(z, (-1, 1, int(self.outsize/8), int(self.outsize/8)))
        z = self.layer2(z)
        z = self.elu(z)
        z = self.layer3(z)
        z = self.elu(z)
        z = self.layer4(z)
        loc_img = self.sigmoid(z)

        return loc_img
