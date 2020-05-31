import os
from load_gz_data import Gz2_data
import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

class Decoder(nn.Module):
    def __init__(self, hidden_dim=200, z_dim=100,
                 height=424, width=424, channels=3):
        super().__init__()
        self.fc4 = nn.Linear(z_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, height)
        self.fc2 = nn.Linear(height, height*width)
        self.fc1 = nn.Linear(height*width, height*width*channels)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = softplus(self.fc4(z))
        z = self.fc3(z)
        z = self.fc2(z)
        z = self.fc1(z)
        z = self.sigmoid(z)
        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

class Encoder(nn.Module):
    def __init__(self, hidden_dim=200, z_dim=50,
                 height=424, width=424, channels=3):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.fc1 = nn.Linear(height*width*channels, height*width)
        self.fc2 = nn.Linear(height*width, height)
        self.fc3 = nn.Linear(height, hidden_dim)
        self.fc41 = nn.Linear(hidden_dim, z_dim)
        self.fc42 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.reshape(-1, self.height*self.width*self.channels)
        x = self.softplus(self.fc1(x))
        x = self.softplus(self.fc2(x))
        x = self.softplus(self.fc3(x))
        z_loc = self.fc41(x)
        z_scale = torch.exp(self.fc42(x))
        return z_loc, z_scale

class VAE(nn.Module):
    def __init__(self, z_dim=50, hidden_dim=200, use_cuda=False):
        super().__init__()
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    
    
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            loc_img = self.decoder.forward(z)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img



if __name__ == "__main__":
    # test vae
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    
    data = Gz2_data(csv_file="gz2_4.csv",
                    root_dir="~/diss/gz2_mini",
                    list_of_interest=[a01,
                                      a02,
                                      a03])
    sample_of_data = data[1]
    # image or data    
    one_image = sample_of_data['image']
    vae = VAE()
    z_loc, z_scale = vae.encoder(one_image)
    out = vae.decoder(z_loc)
