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

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

class Encoder(nn.Module):
    """
    Will take any insize as long as it is divisible by 8
    """
    def __init__(self,
                 insize=56,
                 z_dim=10):
        super().__init__()
        self.insize = insize
        self.linear_size = int((insize/8)**2)
        self.layer1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.layer2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.layer3 = nn.Conv2d(6, 1, 3, 1, 1)
        self.layer41 = nn.Linear(self.linear_size, z_dim)
        self.layer42 = nn.Linear(self.linear_size, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
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
        self.layer2 = nn.ConvTranspose2d(1, 3, 3, 2, 1)
        self.layer3 = nn.ConvTranspose2d(3, 6, 3, 2)
        self.layer4 = nn.ConvTranspose2d(6, 1, 4, 2)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        import pdb
        pdb.set_trace()
        z = self.softplus(self.layer1(z))
        z = torch.reshape(z, (-1, 1, int(self.outsize/8), int(self.outsize/8)))
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.sigmoid(z)
        hidden = self.softplus(z)
        loc_img = self.sigmoid(z)
        return loc_img


class VAE(nn.Module):
    def __init__(self, z_dim=200, hidden_dim=200, use_cuda=False):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
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
    def sample_img(self, x, use_cuda=False, encoder=False, decoder=False):
        # encode image x
        if use_cuda == True:
            x = x.cuda()
        batch_shape = x.shape[0]
        if encoder == False:
            z_loc, z_scale = self.encoder(x)
        else:
            z_loc, z_scale = encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        if decoder == False:
            loc_img = self.decoder(z)
        else:
            loc_img = decoder(z)
        return loc_img.reshape([batch_shape, 1, 424, 424])

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
    csv = "gz2_data/gz2_20.csv"
    img = "gz2_data/"

    parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
    parser.add_argument('--img_file', metavar='i', type=str, default=img)

    args = parser.parse_args()
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    data = Gz2_data(csv_dir=args.csv_file,
                    image_dir=args.img_file,
                    list_of_interest=[a01,
                                      a02,
                                      a03],
                    crop=56,
                    resize=56
                    
    )
    
    # one pass through vae
    dataloader = DataLoader(data, batch_size=2)
    one_image = next(iter(dataloader))['image']
    vae = VAE()
    z_loc, z_scale = vae.encoder(one_image)
    out = vae.decoder(z_loc)
    # run:
    # python gz_vae_conv.py ~/diss/gz2_data/gz_amended.csv /Users/Mizunt/diss/gz2_data
