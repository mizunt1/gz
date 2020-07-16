import math
import torch
from torch.nn import functional as F
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
from kornia import augmentation
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import glob


def transforming_template_mnist(data, transforms, cond=True, grid_size=40, **kwargs):
    with pyro.plate(data.shape[0]):
        # firsnt attempt - literally try to learn a template to check
        # we got the tranformation logic correct
        template = pyro.param(
            "template",
            torch.rand(1, 40, 40, device=data.device),
            constraint=constraints.unit_interval,
        )

        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, grid_size, device=data.device),
                torch.linspace(-1, 1, grid_size, device=data.device),
            ),
            -1,
        )
        grid = grid.expand(data.shape[0], *grid.shape)

        transform = random_pose_transform(transforms)

        transform_grid = transform(grid)

        transformed_template = T.broadcasting_grid_sample(
            template.expand(data.shape[0], 1, 40, 40), transform_grid
        )
        obs = data if cond else None
        pyro.sample("pixels", D.Bernoulli(transformed_template).to_event(3), obs=obs)


def transforming_vae_mnist(
    data, transforms, decoder=None, cond=True, grid_size=28, **kwargs
):
    decoder = pyro.module("decoder", decoder)
    with pyro.plate(data.shape[0]):
        # firsnt attempt - literally try to learn a template to check
        # we got the tranformation logic correct
        template = pyro.param(
            "template",
            torch.rand(1, 32, 32, device=data.device),
            constraint=constraints.unit_interval,
        )

        grid = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, grid_size, device=data.device),
                torch.linspace(-1, 1, grid_size, device=data.device),
            ),
            -1,
        )
        grid = grid.expand(data.shape[0], *grid.shape)

        transform = random_pose_transform(transforms)

        transform_grid = transform(grid)

        transformed_template = T.broadcasting_grid_sample(
            template.expand(data.shape[0], 1, 32, 32), transform_grid
        )
        obs = data if cond else None
        pyro.sample("pixels", D.Bernoulli(transformed_template).to_event(3), obs=obs)


def transforming_template_encoder(data, *args, encoder=None, **kwargs):
    encoder = pyro.module("enc", encoder)

    with pyro.plate(data.shape[0]):
        transform_output = encoder(data)
        delta_sample_transformer_params(
            encoder.transformers, transform_output["params"]
        )


if __name__ == "__main__":

    mnist = MNIST("./data", download=True)
    x_train = mnist.data[mnist.targets == 2][0].float().cuda() / 255
    x_train = x_train[None, None, ...]

    transforms = T.TransformSequence(T.Rotation())

    encoder = transformers.TransformerSequence(
        transformers.Rotation(networks.EquivariantPosePredictor, 1, 32)
    )
    encoder = encoder.cuda()

    opt = optim.Adam({}, clip_args={"clip_norm": 10.0})
    elbo = infer.Trace_ELBO()

    svi = infer.SVI(
        transforming_template_mnist, transforming_template_encoder, opt, loss=elbo
    )
    files = glob.glob("runs/*")
    for f in files:
        os.remove(f)
    tb = SummaryWriter("runs/")

    x_train = x_train.expand(516, 1, 28, 28)
    x_train = F.pad(x_train, (6, 6, 6, 6))
    for i in range(20000):
        x_rot = augmentation.RandomAffine(40.0)(
            x_train
        )  # randomly translate the image by up to 40 degrees

        l = svi.step(x_rot, transforms, cond=True, grid_size=40, encoder=encoder)
        tb.add_scalar("loss", l, i)
        if (i % 100) == 0:
            print(i, l)
            x_rot = x_rot[:32]
            tb.add_image("originals", torchvision.utils.make_grid(x_rot), i)
            bwd_trace = poutine.trace(transforming_template_encoder).get_trace(
                x_rot, transforms, cond=True, grid_size=40, encoder=encoder
            )
            fwd_trace = poutine.trace(
                poutine.replay(transforming_template_mnist, trace=bwd_trace)
            ).get_trace(x_rot, transforms, cond=False, grid_size=40, encoder=encoder)
            recon = fwd_trace.nodes["pixels"]["fn"].mean
            tb.add_image("recons", torchvision.utils.make_grid(recon), i)
