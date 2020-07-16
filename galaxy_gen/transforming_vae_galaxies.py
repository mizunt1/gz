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


TRANSFORM_STRING_MAP = {
    "HyperbolicRotation": T.HyperbolicRotation,
    "PerspectiveX": T.PerspectiveX,
    "PerspectiveY": T.PerspectiveY,
    "Rotation": T.Rotation,
    "RotationScale": T.RotationScale,
    "Scale": T.Scale,
    "ScaleX": T.ScaleX,
    "ScaleY": T.ScaleY,
    "ShearX": T.ShearX,
    "ShearY": T.ShearY,
    "Translation": T.Translation,
}

TRANSFORMER_STRING_MAP = {
    "HyperbolicRotation": transformers.HyperbolicRotation,
    "PerspectiveX": transformers.PerspectiveX,
    "PerspectiveY": transformers.PerspectiveY,
    "Rotation": transformers.Rotation,
    "RotationScale": transformers.RotationScale,
    "Scale": transformers.Scale,
    "ScaleX": transformers.ScaleX,
    "ScaleY": transformers.ScaleY,
    "ShearX": transformers.ShearX,
    "ShearY": transformers.ShearY,
    "Translation": transformers.Translation,
}


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
    use_deltas = False
    kl_beta = 1.0
    transform_sequence = ["Rotation"]
    transformer_features = 128
    oversize_view=False


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
        transform_output = self.transformer(x)
        output["transform"] = transform_output["transform"]
        output["transform_params"] = transform_output["params"]

        grid = coordinates.identity_grid([128, 128], device=x.device)
        grid = grid.expand(x.shape[0], *grid.shape)

        transformed_grid = output["transform"][-1](grid)
        view = T.broadcasting_grid_sample(x, transformed_grid)
        output["view"] = view
        # normalise view. Do this after the transformer to make
        # the padding artifacts less severe.
        view = view - 0.2222
        view = view / 0.156
        out = self.view_encoder(view)
        z_mu, z_std = torch.split(out, self.latent_dim, -1)
        output["z_mu"] = z_mu
        output["z_std"] = z_std
        return output


def forward_model(
    data,
    transforms=None,
    instantiate_label=False,
    cond=True,
    decoder=None,
    output_size=128,
    device=torch.device("cpu"),
    kl_beta=1.0,
    **kwargs,
):
    decoder = pyro.module("view_decoder", decoder)
    N = data.shape[0]
    with poutine.scale_messenger.ScaleMessenger(1 / N):
        with pyro.plate("batch", N):
            with poutine.scale_messenger.ScaleMessenger(kl_beta):
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
            scale = view.shape[-1] / output_size
            grid = grid * (1/scale) # rescales the image co-ordinates so one pixel of the recon corresponds to 1 pixel of the view.

            transform = random_pose_transform(transforms, device=device)

            transform_grid = transform(grid)

            transformed_view = T.broadcasting_grid_sample(view, transform_grid)
            obs = data if cond else None
            pyro.sample("pixels", D.Laplace(transformed_view, 0.5).to_event(3), obs=obs)


@ex.capture
def backward_model(data, *args, encoder=None, use_deltas=False, kl_beta=1.0, **kwargs):
    encoder = pyro.module("encoder", encoder)
    N = data.shape[0]
    with poutine.scale_messenger.ScaleMessenger(1 / N):
        with pyro.plate("batch", N):
            encoder_out = encoder(data)
            delta_sample_transformer_params(
                encoder.transformer.transformers, encoder_out["transform_params"]
            )

            pyro.deterministic("attention_input", encoder_out["view"])

            def make_dist():
                if use_deltas:
                    return D.Delta(encoder_out["z_mu"])
                else:
                    return D.Normal(
                        encoder_out["z_mu"], torch.exp(encoder_out["z_std"]) + 1e-6
                    )

            with poutine.scale_messenger.ScaleMessenger(kl_beta):
                z = pyro.sample("z", make_dist().to_event(1))


@ex.capture
def get_datasets(train_set_size=10000, test_set_size=1000):
    dataset = gz_dataset.GZDataset(
        "/scratch-ssd/oatml/data/gz2",
        transform=tvt.Compose(
            [
                tvt.RandomRotation(180, resample=PIL.Image.BILINEAR),
                tvt.CenterCrop(180),
                tvt.Resize(128),
                tvt.Grayscale(),
                tvt.ToTensor(),
            ]
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
    elif opt_name == "RMSprop":
        return pyro.optim.RMSprop
    raise NotImplementedError


@ex.capture
def transform_sequence(transform_sequence=["Rotation"]):
    return (TRANSFORM_STRING_MAP[x]() for x in transform_sequence)


@ex.capture
def transformer_sequence(transform_sequence=["Rotation"], transformer_features=128):
    return (
        TRANSFORMER_STRING_MAP[x](
            networks.EquivariantPosePredictor, 1, transformer_features
        )
        for x in transform_sequence
    )


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
    kl_beta,
    oversize_view
):
    # pyro.enable_validation(True)

    ds_train, ds_test = get_datasets()
    train_dl = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, num_workers=4, shuffle=True
    )
    test_dl = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=4)

    transforms = T.TransformSequence(*transform_sequence())

    trs = transformers.TransformerSequence(*transformer_sequence())

    encoder = TransformingEncoder(trs, latent_dim=latent_dim)
    encoder = encoder.cuda()
    decoder = VaeResViewDecoder(latent_dim=latent_dim, oversize_output=oversize_view)
    decoder.cuda()

    svi_args = {
        "encoder": encoder,
        "decoder": decoder,
        "instantiate_label": True,
        "transforms": transforms,
        "cond": True,
        "output_size": 128,
        "device": torch.device("cuda"),
        "kl_beta": kl_beta,
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

    def batch_train(engine, batch):
        x = batch[0]

        x = x.cuda()

        l = svi.step(x, **svi_args)
        return l

    train_engine = Engine(batch_train)

    @torch.no_grad()
    def batch_eval(engine, batch):
        x = batch[0]

        x = x.cuda()

        l = svi.evaluate_loss(x, **svi_args)

        # get predictive distribution over y.
        return {"loss": l}

    eval_engine = Engine(batch_eval)

    @eval_engine.on(Events.EPOCH_STARTED)
    def switch_eval_mode(*args):
        print("MODELS IN EVAL MODE")
        encoder.eval()
        decoder.eval()

    @train_engine.on(Events.EPOCH_STARTED)
    def switch_train_mode(*args):
        print("MODELS IN TRAIN MODE")
        encoder.train()
        decoder.train()

    metrics.Average().attach(train_engine, "average_loss")
    metrics.Average(output_transform=lambda x: x["loss"]).attach(
        eval_engine, "average_loss"
    )

    @eval_engine.on(Events.EPOCH_COMPLETED)
    def log_tboard(engine):
        ex.log_scalar(
            "train.loss",
            train_engine.state.metrics["average_loss"],
            train_engine.state.epoch,
        )
        ex.log_scalar(
            "eval.loss",
            eval_engine.state.metrics["average_loss"],
            train_engine.state.epoch,
        )
        tb.add_scalar(
            "train/loss",
            train_engine.state.metrics["average_loss"],
            train_engine.state.epoch,
        )
        tb.add_scalar(
            "eval/loss",
            eval_engine.state.metrics["average_loss"],
            train_engine.state.epoch,
        )

        print(
            "EPOCH",
            train_engine.state.epoch,
            "train.loss",
            train_engine.state.metrics["average_loss"],
            "eval.loss",
            eval_engine.state.metrics["average_loss"],
        )

    def plot_recons(dataloader, mode):
        epoch = train_engine.state.epoch
        for batch in dataloader:
            x = batch[0]
            x = x.cuda()
            break
        x = x[:64]
        tb.add_image(f"{mode}/originals", torchvision.utils.make_grid(x), epoch)
        bwd_trace = poutine.trace(backward_model).get_trace(x, **svi_args)
        fwd_trace = poutine.trace(
            poutine.replay(forward_model, trace=bwd_trace)
        ).get_trace(x, **svi_args)
        recon = fwd_trace.nodes["pixels"]["fn"].mean
        tb.add_image(f"{mode}/recons", torchvision.utils.make_grid(recon), epoch)

        canonical_recon = fwd_trace.nodes["canonical_view"]["value"]
        tb.add_image(
            f"{mode}/canonical_recon",
            torchvision.utils.make_grid(canonical_recon),
            epoch,
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
            f"{mode}/prior_samples", torchvision.utils.make_grid(prior_sample), epoch
        )

        tb.add_image(
            f"{mode}/canonical_prior_samples",
            torchvision.utils.make_grid(prior_canonical_sample),
            epoch,
        )
        tb.add_image(
            f"{mode}/input_view",
            torchvision.utils.make_grid(bwd_trace.nodes["attention_input"]["value"]),
            epoch,
        )

    @eval_engine.on(Events.EPOCH_COMPLETED)
    def plot_images(engine):
        plot_recons(train_dl, "train")
        plot_recons(test_dl, "eval")

    @train_engine.on(Events.EPOCH_COMPLETED(every=eval_every))
    def eval(engine):
        eval_engine.run(test_dl, seed=_seed + engine.state.epoch)

    train_engine.run(train_dl, max_epochs=max_epochs, seed=_seed)
