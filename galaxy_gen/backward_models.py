import pyro
import torch
from pyro import distributions as D
from torch import nn
from torch.nn import functional as F


def delta_sample_transformer_params(transformers, params):
    for t, p in zip(transformers, params):
        sample_site_prefix = repr(type(t.transform)).split(".")[-1].strip("'>")

        u_sample_name = sample_site_prefix + "_u"
        v_sample_name = sample_site_prefix + "_v"

        if t.transform.has_u and t.transform.has_v:
            u, v = p
            # periodic constraints are taken care of by the prior
            u = pyro.sample(u_sample_name, D.Delta(u))

            v = pyro.sample(v_sample_name, D.Delta(v))
        elif t.transform.has_u:
            u = p[0]
            u = pyro.sample(u_sample_name, D.Delta(u))
        elif t.transform.has_v:

            v = p[0]
            v = pyro.sample(v_sample_name, D.Delta(v))


class UpResBloc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.skip = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.merge(self.skip(x) + self.body(x))


class DownResBloc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.MaxPool2d(2),
        )
        self.skip = nn.MaxPool2d(scale_factor=2, mode="nearest")
        self.merge = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.merge(self.skip(x) + self.body(x))


class VaeResViewDecoder(nn.Module):
    def __init__(self, latent_dim=1024, oversize_output=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_to_hidden = nn.Linear(self.latent_dim, 32 * 8 * 8)
        if oversize_output:
            self.fct_decode = nn.Sequential(
                UpResBloc(32, 64),  # 8-> 16
                UpResBloc(64, 64),  # 16 -> 32
                UpResBloc(64, 64),  # 32 -> 64
                UpResBloc(64, 16),  # 64 -> 128
                UpResBloc(16, 16),  # 128 -> 256
                nn.ELU(),
                nn.Conv2d(16, 1, (3, 3), padding=1),
                nn.Sigmoid(),
            )
        else:
            self.fct_decode = nn.Sequential(
                UpResBloc(32, 64),  # 8-> 16
                UpResBloc(64, 64),  # 16 -> 32
                UpResBloc(64, 64),  # 32 -> 64
                UpResBloc(64, 16),  # 64 -> 128
                nn.ELU(),
                nn.Conv2d(16, 1, (3, 3), padding=1),
                nn.Sigmoid(),
            )

    def forward(self, z):
        z = F.elu(self.latent_to_hidden(z))
        z = z.view(z.shape[0], 32, 8, 8)
        x = self.fct_decode(z)
        return x


class VaeResViewEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            Conv_Block(1, 64, (3, 3), 1, 1),  # 64
            nn.ELU(),
            Conv_Block(64, 128, (3, 3), 1, 1),  # 32
            nn.ELU(),
            Conv_Block(128, 256, (3, 3), 1, 1),  # 16
            nn.ELU(),
            Conv_Block(256, 32, (3, 3), 1, 1),  # 8
        )
        self.hidden_to_latent = nn.Linear(32 * 8 * 8, self.latent_dim * 2)

    def forward(self, x):
        z = F.elu(self.encoder(x))
        z = torch.flatten(z, 1, -1)
        z = self.hidden_to_latent(z)
        return z


class VAEViewDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 1024
        self.fct_decode = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 64
            nn.Conv2d(64, 16, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 128
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = z.view(z.shape[0], 16, 8, 8)
        x = self.fct_decode(z)
        return x


class Conv_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        pool_kernel_size=(2, 2),
    ):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)

        return x


class VAEViewEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv_Block(1, 64, (3, 3), 1, 1),  # 64
            nn.ELU(),
            Conv_Block(64, 128, (3, 3), 1, 1),  # 32
            nn.ELU(),
            Conv_Block(128, 256, (3, 3), 1, 1),  # 16
            nn.ELU(),
            Conv_Block(256, 32, (3, 3), 1, 1),  # 8
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.shape[0], -1)
        return z


class DCGanViewDecoder(nn.Module):
    """
    View decoder, based on DCGAN
    """

    def __init__(self, nz=32, ngf=64, nc=1):
        super().__init__()
        self.latent_dim = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        z = z[:, :, None, None]
        return self.main(z)


class DCGanViewEncoder(nn.Module):
    def __init__(self, nz=32, nc=1, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )
        self.view_to_latent = nn.Linear(ndf * 8 * 4 * 4, nz)

    def forward(self, input):
        z = self.main(input)
        z = torch.flatten(z, 1, -1)
        return self.view_to_latent(z)
