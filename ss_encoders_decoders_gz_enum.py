import torch.nn as nn
import torch
import utils 
from layers import ConvBlock, UpResBloc, Conv2dEnum, BatchNorm2dEnum
class Encoder_z(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    """
    def __init__(self,
                 x_size=80, z_size=100, y_size=3):
        super().__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.z_size = z_size
        self.linear_size = int((self.x_size/8)**2)
        self.net = nn.Sequential(
            Conv2dEnum(1, 32, kernel_size=7, padding=3, bias=False),
            nn.ELU(),
            nn.AvgPool2d(2),
            ConvBlock(32, bias=False),
            Conv2dEnum(32,16,kernel_size=5, padding=2, bias=False),
            nn.ELU(),
            ConvBlock(16, bias=False),
            nn.AvgPool2d(2),
            Conv2dEnum(16, 1, kernel_size=5, padding=2, bias=False),
            nn.AvgPool2d(2),
            nn.ELU()
        )
        self.linear = nn.Linear(self.linear_size + self.y_size, self.linear_size)
        self.loc = nn.Linear(self.linear_size, self.z_size)
        self.scale = nn.Linear(self.linear_size, self.z_size)
        
    def forward(self, x, y):
        z = x - 0.222
        z = x / 0.156
        z = self.net(z)
        z = z.view(z.shape[0], -1)
        
        z = utils.cat((z, y), -1)

        z = self.linear(z)
        z_loc = self.loc(z)
        z_scale = torch.exp(self.scale(z))
        return z_loc, z_scale

class Encoder_y(nn.Module):
    def __init__(self, x_size=80, y_size=3):
        super().__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.linear_size = int((x_size/8)**2)
        self.net = nn.Sequential(
            Conv2dEnum(1, 32, kernel_size=7, padding=3, bias=False),
            nn.Tanh(),
            nn.AvgPool2d(2),
            ConvBlock(32,5, bias=False),
            Conv2dEnum(32,16,kernel_size=5, padding=2, bias=False),
            nn.Tanh(),
            ConvBlock(16, bias=False),
            nn.AvgPool2d(2),
            Conv2dEnum(16, 1, kernel_size=5, padding=2, bias=False),
            nn.AvgPool2d(2),
            nn.Tanh()
        )
        self.linear = nn.Linear(self.linear_size, self.y_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        x = x - 0.222
        x = x / 0.156
        x = self.net(x)
        x = x.view(-1, self.linear_size)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, z_size=100, x_size=80, y_size=3):
        super().__init__()
        self.y_size = y_size
        self.z_size = z_size
        self.x_size = x_size
        self.linear_size = int((x_size/8)**2)
        self.linear = nn.Linear(self.z_size + self.y_size, self.linear_size)
        self.net = nn.Sequential(
            nn.ELU(),
            UpResBloc(1, 32),
            nn.ELU(),
            BatchNorm2dEnum(32),
            ConvBlock(32, bias=False),
            UpResBloc(32, 32),
            nn.ELU(),
            ConvBlock(32, bias=False),
            ConvBlock(32, bias=False),
            UpResBloc(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z, y):
        # TODO CHECK
        z = utils.cat((z, y), -1)
        z = self.linear(z)
        z = torch.reshape(z, (*z.shape[:-1],1, int(self.x_size/8), int(self.x_size/8)))
        
        loc_img = self.net(z)
        return loc_img

if __name__ == "__main__":
    x = torch.zeros([10, 1, 80, 80])
    y = torch.zeros([2, 10, 3])
    encoder_z = Encoder_z(x_size=80)
    encoder_y = Encoder_y()
    decoder = Decoder(x_size=80)
    x = encoder_z(x, y)

    x = x[0]
    x = decoder(x, y)
    x = torch.zeros([10, 1, 80, 80])
    y = encoder_y(x)
