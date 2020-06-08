import os
from load_gz_data import Gz2_data
import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', metavar='c', type=str)
parser.add_argument('img_file', metavar='i', type=str)

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

class Encoder(nn.Module):
    def __init__(self, hidden_dim=200, z_dim=100,
                 height=424, width=424, channels=3):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.layer1 = nn.Conv2d(1, 3, 6, 2)
        self.layer2 = nn.Conv2d(3, 6, 5, 2)
        self.layer3 = nn.Conv2d(6, 1, 5, 2)
        self.layer41 = nn.Linear(2500, z_dim)
        self.layer42 = nn.Linear(2500, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1, 2500)
        z_loc = self.layer41(x)
        z_scale = torch.exp(self.layer42(x))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, hidden_dim=200, z_dim=100,
                 height=424, width=424, channels=3):
        super().__init__()
        self.layer4 = nn.Linear(z_dim, 2500)
        self.layer3 = nn.ConvTranspose2d(1, 3, 5, 2, output_padding=1)
        self.layer2 = nn.ConvTranspose2d(3, 6, 5, 4, output_padding=2)
        self.layer1 = nn.ConvTranspose2d(6, 1, 6, 1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.softplus(self.layer4(z))
        z = z.reshape(-1, 1, 50, 50)
        z = self.layer3(z)
        z = self.layer2(z)
        z = self.layer1(z)
        z = self.sigmoid(z)
        hidden = self.softplus(z)
        loc_img = self.sigmoid(z)
        return loc_img


class VAE(nn.Module):
    def __init__(self, z_dim=200, hidden_dim=200, use_cuda=False):
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
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            # decoder is where the image goes
            # is this correct?
            # Ask lewis, channel, and h and w are dependent, go to event
            pyro.sample(
                "obs",
                dist.Bernoulli(loc_img).to_event(3),
                obs=x)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

vae = VAE()

optimizer = Adam({"lr": 1.0e-3})

svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

def train(svi, train_loader, use_cuda=False):
    epoch_loss = 0.
    for x in train_loader:
        x = x['image']
        if use_cuda:
            x = x.cuda()
        epoch_loss += svi.step(x)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    test_loss = 0.
    for x in test_loader:
        x = x['image']
        if use_cuda:
            x = x.cuda()
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


if __name__ == "__main__":
    # test vae
    args = parser.parse_args()
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    data = Gz2_data(csv_dir=args.csv_file,
                    image_dir=args.img_file,
                    list_of_interest=[a01,
                                      a02,
                                      a03])
    
    # one pass through vae
    dataloader = DataLoader(data, batch_size=2)
    one_image = next(iter(dataloader))['image']
    vae = VAE()
    z_loc, z_scale = vae.encoder(one_image)
    out = vae.decoder(z_loc)
    # run:
    # python gz_vae_conv.py ~/diss/gz2_data/gz_amended.csv /Users/Mizunt/diss/gz2_data
