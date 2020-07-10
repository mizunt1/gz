import torch
import torch.nn as nn
import math
import torch
from torch.nn import functional as F
from torch import nn
from torch.distributions import constraints
from torchvision.datasets import MNIST
from torchvision import transforms as tvt
import pyro
from pyro import infer, optim, poutine
from pyro import distributions as D
from galaxy-gen.forward_models import random_pose_transform
from galaxy-gen.backward_models import delta_sample_transformer_params
from galaxy-gen.etn import transformers, networks
from galaxy-gen.etn import transforms as T
from galaxy-gen.etn import coordinates
from kornia import augmentation
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import glob


class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    output is a dictionary with four variables
    dictionary["transform"] is a transformation
    dctionary["transform_params"] are parameters that parameterise that 
    transformation
    dictionary["z_mu"] is the mean of the latent variable
    dictionary["std_mu"] is the standard dev of that latent variable
    so the latent space is the z and std in this transformed coordinate frame
    We output the theta that parameterises this coordinate frame
    and the z in this transformed coordinate frame

    """
    def __init__(self,
                 transformer: transformers.Transformer,
                 insize=56, z_dim=10):
        super().__init__()
        self.transformer = transformer
        self.insize = insize
        self.linear_size = int((insize/8)**2)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.ELU(),
            nn.AvgPool2d(2),
            ConvBlock(32, bias=False),
            nn.Conv2d(32,16,kernel_size=5, padding=2, bias=False),
            nn.ELU(),
            ConvBlock(16, bias=False),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 1, kernel_size=5, padding=2, bias=False),
            nn.AvgPool2d(2),
            nn.ELU()
        )
        self.loc = nn.Linear(self.linear_size, z_dim)
        self.scale = nn.Linear(self.linear_size, z_dim)
        
    def forward2(self, x):
        
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        z_loc = self.loc(x)
        z_scale = torch.exp(self.scale(x))
        return z_loc, z_scale

    def forward(self, x):
        output = {}
        transform_output = self.transformer(x)
        output["transform"] = transform_output["transform"]
        output["transform_params"] = transform_output["params"]
        
        grid = coordinates.identity_grid(
            [self.input_size, self.input_size], device=x.device
        )
        grid = grid.expand(x.shape[0], *grid.shape)

        transformed_grid = output["transform"][-1](grid)
        
        x = x - 0.222
        x = x / 0.156

        view = T.broadcasting_grid_sample(x, transformed_grid)
        
        out = self.encoder(view)
        out = out.view(out.shape[0], -1)
        z_loc = self.loc(out)
        z_scale = torch.exp(self.scale(out))
        output["z_mu"] = z_loc
        output["z_std"] = z_scale
        
        return output

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
    x = torch.zeros([10, 1, 80, 80])
    encoder = Encoder(insize=80)
    decoder = Decoder(outsize=80)
    x = encoder(x)
    x = x[0]
    x = decoder(x)

    
