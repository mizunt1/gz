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
from forward_models import random_pose_transform
from backward_models import delta_sample_transformer_params
from etn import transformers, networks
from etn import transforms as T
from etn import coordinates
from kornia import augmentation
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import glob


class ViewDecoder(nn.Module):
    def __init__(self, grid_size=30, latent_dim=32, num_channels=1):

        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.num_channels = 1

        self.view_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.ELU(),
            nn.Linear(self.latent_dim * 2, self.latent_dim * 4),
            nn.ELU(),
            nn.Linear(self.latent_dim * 4, self.latent_dim * 8),
            nn.ELU(),
            nn.Linear(self.latent_dim * 8, self.grid_size ** 2 * self.num_channels),
            nn.Sigmoid(),
        )

    def forward(self, z):
        view = self.view_decoder(z)
        view = view.view(
            *z.shape[:-1], self.num_channels, self.grid_size, self.grid_size
        )
        return view


class TransformingEncoder(nn.Module):
    def __init__(
        self,
        transformer: transformers.Transformer,
        input_size=40,
        latent_dim=32,
        num_channels=1,
    ):
        super().__init__()
        self.transformer = transformer
        self.input_size = input_size
        self.latent_dim = latent_dim
        out_size = math.floor(self.input_size - 4 + 1) // 2
        out_size = math.floor(out_size - 4 + 1) // 2
        self.view_encoder = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3),
            nn.ELU(),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 128, 3),
            nn.ELU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * out_size ** 2, self.latent_dim * 2),
        )

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
        view = T.broadcasting_grid_sample(x, transformed_grid)
        out = self.view_encoder(view)
        z_mu, z_std = torch.split(out, self.latent_dim, -1)
        output["z_mu"] = z_mu
        output["z_std"] = z_std

        return output


def forward_model(
    data,
    transforms=None,
    cond=True,
    decoder=None,
    output_size=40,
    device=torch.device("cpu"),
    **kwargs
):
    decoder = pyro.module("view_decoder", decoder)
    with pyro.plate(data.shape[0]):

        z = pyro.sample(
            "z",
            D.Normal(
                torch.zeros(decoder.latent_dim, device=device),
                torch.ones(decoder.latent_dim, device=device),
            ).to_event(1),
        )

        view = decoder(z)

        pyro.deterministic("canonical_view", view)

        grid = coordinates.identity_grid([output_size, output_size], device=device)
        grid = grid.expand(data.shape[0], *grid.shape)

        transform = random_pose_transform(transforms)

        transform_grid = transform(grid)

        transformed_view = T.broadcasting_grid_sample(view, transform_grid)
        obs = data if cond else None
        pyro.sample("pixels", D.Bernoulli(transformed_view).to_event(3), obs=obs)


def backward_model(data, encoder=None, **kwargs):
    encoder = pyro.module("encoder", encoder)
    with pyro.plate(data.shape[0]):
        encoder_out = encoder(data)
        delta_sample_transformer_params(
            encoder.transformer.transformers, encoder_out["transform_params"]
        )

        z = pyro.sample(
            "z",
            D.Normal(
                encoder_out["z_mu"], torch.exp(encoder_out["z_std"]) + 1e-3
            ).to_event(1),
        )


if __name__ == "__main__":

    mnist = MNIST(
        "./data",
        download=True,
        transform=tvt.Compose(
            [
                tvt.Pad(6),
                tvt.RandomAffine(degrees=180, translate=(0.12, 0.12)),
                tvt.ToTensor(),
            ]
        ),
    )
    train_dl = torch.utils.data.DataLoader(mnist, batch_size=256)

    transforms = T.TransformSequence(T.Translation(), T.Rotation())

    transformers = transformers.TransformerSequence(
        transformers.Translation(networks.EquivariantPosePredictor, 1, 32),
        transformers.Rotation(networks.EquivariantPosePredictor, 1, 32),
    )
    encoder = TransformingEncoder(transformers)
    encoder = encoder.cuda()
    decoder = ViewDecoder(grid_size=32)
    decoder.cuda()

    svi_args = {
        "encoder": encoder,
        "decoder": decoder,
        "transforms": transforms,
        "cond": True,
        "output_size": 40,
        "device": torch.device("cuda"),
    }

    opt = optim.Adam({}, clip_args={"clip_norm": 10.0})
    elbo = infer.Trace_ELBO()

    svi = infer.SVI(forward_model, backward_model, opt, loss=elbo)
    files = glob.glob("runs/*")
    for f in files:
        os.remove(f)
    tb = SummaryWriter("runs/")

    for epoch in range(100):
        losses = []
        for (x, _) in train_dl:
            x = x.cuda()

            l = svi.step(x, **svi_args)
            losses.append(l)

        l = sum(losses) / len(losses)
        tb.add_scalar("loss", l, epoch)

        print(epoch, l)
        x = x[:32]
        tb.add_image("originals", torchvision.utils.make_grid(x), epoch)
        bwd_trace = poutine.trace(backward_model).get_trace(x, **svi_args)
        fwd_trace = poutine.trace(
            poutine.replay(forward_model, trace=bwd_trace)
        ).get_trace(x, **svi_args)
        recon = fwd_trace.nodes["pixels"]["fn"].mean
        tb.add_image("recons", torchvision.utils.make_grid(recon), epoch)

        canonical_recon = fwd_trace.nodes["canonical_view"]["value"]
        tb.add_image(
            "canonical_recon", torchvision.utils.make_grid(canonical_recon), epoch
        )
