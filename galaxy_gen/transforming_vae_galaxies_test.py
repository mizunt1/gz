import glob
import math
import os
from pathlib import Path

import PIL
import torch
import torchvision
from ignite import metrics
from ignite.engine import Engine, Events
from kornia import augmentation
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.distributions import constraints
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tvt

import gz_dataset
import pyro
import utils as U
from backward_models import (
    VaeResViewEncoder,
    VaeResViewDecoder,
    delta_sample_transformer_params,
)
from etn import coordinates, networks, transformers
from etn import transforms as T
from forward_models import random_pose_transform
from pyro import distributions as D
from pyro import infer, optim, poutine


sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append("SLURM_JOB_ID")
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.append("SLURM_JOB_NAME")

FILE_STORAGE_OBSERVER_PATH = "/users/lewith/Documents/sacred_experiments/galaxy-gen"
TENSORBOARD_OBSERVER_PATH = "/users/lewith/Documents/tensorboards/galaxy-gen"
ex = Experiment()
# ex.observers.append(FileStorageObserver(FILE_STORAGE_OBSERVER_PATH))


class TransformingEncoder(nn.Module):
    def __init__(self, transformer: transformers.Transformer, latent_dim=1024):
        super().__init__()
        self.transformer = transformer
        self.view_encoder = VaeResViewEncoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        output = {}
        # apply normalisation.
        # don't want to do this in the dataloader because we want values in [0,1] for the bernoulli loss
        x = x - 0.2222
        x = x / 0.156
        transform_output = self.transformer(x)
        output["transform"] = transform_output["transform"]
        output["transform_params"] = transform_output["params"]

        grid = coordinates.identity_grid([128, 128], device=x.device)
        grid = grid.expand(x.shape[0], *grid.shape)

        transformed_grid = output["transform"][-1](grid)
        view = T.broadcasting_grid_sample(x, transformed_grid)
        out = self.view_encoder(view)
        z_mu, z_std = torch.split(out, self.latent_dim, -1)
        output["z_mu"] = z_mu
        output["z_std"] = z_std
        output["view"] = view
        return output


def forward_model(
    data,
    transforms=None,
    instantiate_label=False,
    cond=True,
    decoder=None,
    output_size=128,
    device=torch.device("cpu"),
    **kwargs,
):
    decoder = pyro.module("view_decoder", decoder)
    N = data.shape[0]
    with poutine.scale_messenger.ScaleMessenger(1 / N):
        with pyro.plate("batch", N):

            z = pyro.sample(
                "z",
                D.Normal(
                    torch.zeros(N, decoder.latent_dim, device=device),
                    torch.ones(N, decoder.latent_dim, device=device),
                ).to_event(1),
            )

            # use supervision

            view = decoder(z)

            pyro.deterministic("canonical_view", view)

            grid = coordinates.identity_grid([output_size, output_size], device=device)
            grid = grid.expand(N, *grid.shape)

            transform = random_pose_transform(transforms, device=device)

            transform_grid = transform(grid)

            transformed_view = T.broadcasting_grid_sample(view, transform_grid)
            obs = data if cond else None
            pyro.sample("pixels", D.Bernoulli(transformed_view).to_event(3), obs=obs)


def backward_model(data, *args, encoder=None, **kwargs):
    encoder = pyro.module("encoder", encoder)
    N = data.shape[0]
    with poutine.scale_messenger.ScaleMessenger(1 / N):
        with pyro.plate("batch", N):
            encoder_out = encoder(data)
            delta_sample_transformer_params(
                encoder.transformer.transformers, encoder_out["transform_params"]
            )

            pyro.deterministic("attention_input", encoder_out["view"])
            z = pyro.sample(
                "z",
                D.Normal(
                    encoder_out["z_mu"], torch.exp(encoder_out["z_std"]) + 1e-3
                ).to_event(1),
            )


@ex.config
def config():
    opt_alg = "Adam"
    opt_args = {"lr": 1e-3}
    clip_args = {"clip_norm": 10}
    max_epochs = 100
    batch_size = 128
    latent_dim = 1024
    eval_every = 1
    train_set_size = 10000
    test_set_size = 1000


@ex.capture
def get_datasets(train_set_size, test_set_size):
    dataset = gz_dataset.GZDataset(
        "/scratch/gz_data/gz2",
        transform=tvt.Compose(
            [tvt.CenterCrop(200), tvt.Resize(128), tvt.Grayscale(), tvt.ToTensor()]
        ),
    )
    train = torch.utils.data.Subset(dataset, range(train_set_size))
    val = torch.utils.data.Subset(
        dataset, range(train_set_size, train_set_size + test_set_size)
    )
    return train, val


def get_opt_alg(opt_name):
    if opt_name == "Adam":
        return pyro.optim.Adam
    raise NotImplementedError


@ex.automain
def main(
    opt_alg,
    opt_args,
    clip_args,
    max_epochs,
    batch_size,
    latent_dim,
    _seed,
    _run,
    eval_every,
):
    # pyro.enable_validation(True)

    ds_train, ds_test = get_datasets()
    train_dl = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, num_workers=4, shuffle=True
    )
    test_dl = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=4)

    transforms = T.TransformSequence(T.Rotation())

    trs = transformers.TransformerSequence(
        transformers.Rotation(networks.EquivariantPosePredictor, 1, 32)
    )

    encoder = TransformingEncoder(trs, latent_dim=latent_dim)
    encoder = encoder.cuda()
    decoder = VaeResViewDecoder(latent_dim=latent_dim)
    decoder.cuda()

    svi_args = {
        "encoder": encoder,
        "decoder": decoder,
        "instantiate_label": True,
        "transforms": transforms,
        "cond": True,
        "output_size": 128,
        "device": torch.device("cuda"),
    }

    opt_alg = get_opt_alg(opt_alg)
    opt = opt_alg(opt_args, clip_args=clip_args)
    elbo = infer.Trace_ELBO(max_plate_nesting=1)

    svi = infer.SVI(forward_model, backward_model, opt, loss=elbo)

    if _run.unobserved or _run._id is None:
        tb = U.DummyWriter("/tmp/delme")
    else:
        tb = SummaryWriter(
            U.setup_run_directory(Path(TENSORBOARD_OBSERVER_PATH) / repr(_run._id))
        )
        _run.info["tensorboard"] = tb.log_dir

    for batch in train_dl:
        x = batch[0]
        x_orig = x.cuda()
        break

    for i in range(10000):
        encoder.train()
        decoder.train()
        x = augmentation.RandomRotation(180.0)(x_orig)
        l = svi.step(x, **svi_args)

        if i % 200 == 0:
            encoder.eval()
            decoder.eval()

            print("EPOCH", i, "LOSS", l)
            ex.log_scalar("train.loss", l, i)
            tb.add_scalar("train/loss", l, i)
            tb.add_image(f"train/originals", torchvision.utils.make_grid(x), i)
            bwd_trace = poutine.trace(backward_model).get_trace(x, **svi_args)
            fwd_trace = poutine.trace(
                poutine.replay(forward_model, trace=bwd_trace)
            ).get_trace(x, **svi_args)
            recon = fwd_trace.nodes["pixels"]["fn"].mean
            tb.add_image(f"train/recons", torchvision.utils.make_grid(recon), i)

            canonical_recon = fwd_trace.nodes["canonical_view"]["value"]
            tb.add_image(
                f"train/canonical_recon",
                torchvision.utils.make_grid(canonical_recon),
                i,
            )

            # sample from the prior

            prior_sample_args = {}
            prior_sample_args.update(svi_args)
            prior_sample_args["cond"] = False
            prior_sample_args["cond_label"] = False
            fwd_trace = poutine.trace(forward_model).get_trace(x, **prior_sample_args)
            prior_sample = fwd_trace.nodes["pixels"]["fn"].mean
            prior_canonical_sample = fwd_trace.nodes["canonical_view"]["value"]
            tb.add_image(
                f"train/prior_samples", torchvision.utils.make_grid(prior_sample), i
            )

            tb.add_image(
                f"train/canonical_prior_samples",
                torchvision.utils.make_grid(prior_canonical_sample),
                i,
            )
            tb.add_image(
                f"train/input_view",
                torchvision.utils.make_grid(
                    bwd_trace.nodes["attention_input"]["value"]
                ),
                i,
            )
