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
from ignite.engine import Engine, Events
from ignite import metrics
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
            nn.LayerNorm([1, 40, 40]),
            nn.Conv2d(num_channels, 32, 3),
            nn.ELU(),
            nn.LayerNorm([32, 38, 38]),
            nn.Conv2d(32, 64, 3),
            nn.ELU(),
            nn.AvgPool2d(2),
            nn.LayerNorm([64, 18, 18]),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            nn.LayerNorm([64, 16, 16]),
            nn.Conv2d(64, 32, 3),
            nn.ELU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * out_size ** 2, self.latent_dim * 2),
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
    label,
    N=-1,
    transforms=None,
    instantiate_label=False,
    cond_label=True,
    cond=True,
    decoder=None,
    latent_decoder=None,
    output_size=40,
    device=torch.device("cpu"),
    **kwargs
):
    decoder = pyro.module("view_decoder", decoder)
    with pyro.plate("batch", N):

        z = pyro.sample(
            "z",
            D.Normal(
                torch.zeros(N, decoder.latent_dim, device=device),
                torch.ones(N, decoder.latent_dim, device=device),
            ).to_event(1),
        )

        # use supervision
        if instantiate_label:
            latent_decoder = pyro.module("latent_decoder", latent_decoder)
            label_logits = latent_decoder(z)
            obs_label = label if cond_label else None
            pyro.sample("y", D.Categorical(logits=label_logits), obs=obs_label)

        view = decoder(z)

        pyro.deterministic("canonical_view", view)

        grid = coordinates.identity_grid([output_size, output_size], device=device)
        grid = grid.expand(N, *grid.shape)

        transform = random_pose_transform(transforms, device=device)

        transform_grid = transform(grid)

        transformed_view = T.broadcasting_grid_sample(view, transform_grid)
        obs = data if cond else None
        pyro.sample("pixels", D.Bernoulli(transformed_view).to_event(3), obs=obs)


def backward_model(data, *args, encoder=None, N=-1, **kwargs):
    encoder = pyro.module("encoder", encoder)
    with pyro.plate("batch", N):
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
    # pyro.enable_validation(True)
    eval_every = 2

    augmentation = tvt.Compose(
        [
            tvt.Pad(6),
            tvt.RandomAffine(degrees=90.0, translate=(0.14, 0.14), scale=(0.8, 1.2)),
            tvt.ToTensor(),
        ]
    )

    mnist = MNIST(
        "./data",
        download=True,
        transform=augmentation,
        # target_transform=target_transform
    )
    mnist_test = MNIST(
        "./data",
        download=True,
        train=False,
        transform=augmentation,
        # target_transform=target_transform
    )
    train_dl = torch.utils.data.DataLoader(mnist, batch_size=128, shuffle=True)
    test_dl = torch.utils.data.DataLoader(mnist_test, batch_size=1000)

    transforms = T.TransformSequence(T.Translation(), T.RotationScale())

    transformers = transformers.TransformerSequence(
        transformers.Translation(networks.EquivariantPosePredictor, 1, 32),
        transformers.RotationScale(networks.EquivariantPosePredictor, 1, 32),
    )
    latent_dim = 64
    encoder = TransformingEncoder(transformers, latent_dim=latent_dim)
    encoder = encoder.cuda()
    decoder = ViewDecoder(grid_size=32, latent_dim=latent_dim)
    decoder.cuda()
    latent_decoder = nn.Sequential(
        nn.Linear(decoder.latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    latent_decoder = latent_decoder.cuda()

    svi_args = {
        "encoder": encoder,
        "decoder": decoder,
        "instantiate_label": True,
        "cond_label": True,
        "latent_decoder": latent_decoder,
        "transforms": transforms,
        "cond": True,
        "output_size": 40,
        "device": torch.device("cuda"),
    }

    opt = optim.Adam({"amsgrad": True}, clip_args={"clip_norm": 10.0})
    elbo = infer.Trace_ELBO(max_plate_nesting=1)

    svi = infer.SVI(forward_model, backward_model, opt, loss=elbo)
    files = glob.glob("runs/*")
    for f in files:
        os.remove(f)
    tb = SummaryWriter("runs/")

    def batch_train(engine, batch):
        x, y = batch

        x = x.cuda()
        y = y.cuda()

        l = svi.step(x, y, N=x.shape[0], **svi_args)
        return l

    train_engine = Engine(batch_train)

    @torch.no_grad()
    def batch_eval(engine, batch):
        x, y = batch

        x = x.cuda()
        y = y.cuda()

        l = svi.evaluate_loss(x, y, N=x.shape[0], **svi_args)

        # get predictive distribution over y.
        eval_args = {}
        eval_args.update(svi_args)

        pred = infer.Predictive(
            forward_model,
            guide=backward_model,
            num_samples=50,
            parallel=True,
            return_sites=["y"],
        )
        pred_trace = pred.get_vectorized_trace(x, y, N=x.shape[0], **eval_args)
        y_probs = pred_trace.nodes["y"]["fn"].probs.mean(0)

        z = pred_trace.nodes["z"]["value"].mean(0)
        # also plot a linear classifier trained offline on z
        train_split = int(x.shape[0] * 0.75)
        z_tr = z[:train_split].cpu().numpy()
        y_tr = y[:train_split].cpu().numpy()
        z_ts = z[train_split:].cpu().numpy()
        y_ts = y[train_split:].cpu().numpy()

        lr = Pipeline(
            [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=10000))]
        )
        lr.fit(z_tr, y_tr)
        acc = lr.score(z_ts, y_ts)
        return {"y": y, "loss": l, "y_pred": y_probs, "y_probs": y_probs, "lr_acc": acc}

    eval_engine = Engine(batch_eval)

    metrics.Accuracy().attach(eval_engine, "accuracy")
    metrics.Average().attach(train_engine, "average_loss")
    metrics.Average(output_transform=lambda x: x["lr_acc"]).attach(
        eval_engine, "lr_acc"
    )
    metrics.Average(output_transform=lambda x: x["loss"]).attach(
        eval_engine, "average_loss"
    )

    @eval_engine.on(Events.EPOCH_COMPLETED)
    def log_tboard(engine):
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
        tb.add_scalar(
            "eval/acc", eval_engine.state.metrics["accuracy"], train_engine.state.epoch
        )
        tb.add_scalar(
            "eval/lr_acc", eval_engine.state.metrics["lr_acc"], train_engine.state.epoch
        )

    @eval_engine.on(Events.EPOCH_COMPLETED)
    def plot_images(engine):
        epoch = train_engine.state.epoch
        for x, y in test_dl:
            x = x.cuda()
            y = y.cuda()
            break
        x = x[:32]
        tb.add_image("originals", torchvision.utils.make_grid(x), epoch)
        bwd_trace = poutine.trace(backward_model).get_trace(
            x, y, N=x.shape[0], **svi_args
        )
        fwd_trace = poutine.trace(
            poutine.replay(forward_model, trace=bwd_trace)
        ).get_trace(x, y, N=x.shape[0], **svi_args)
        recon = fwd_trace.nodes["pixels"]["fn"].mean
        tb.add_image("recons", torchvision.utils.make_grid(recon), epoch)

        canonical_recon = fwd_trace.nodes["canonical_view"]["value"]
        tb.add_image(
            "canonical_recon", torchvision.utils.make_grid(canonical_recon), epoch
        )

        # sample from the prior

        prior_sample_args = {}
        prior_sample_args.update(svi_args)
        prior_sample_args["cond"] = False
        prior_sample_args["cond_label"] = False
        fwd_trace = poutine.trace(forward_model).get_trace(
            x, y, N=x.shape[0], **prior_sample_args
        )
        prior_sample = fwd_trace.nodes["pixels"]["fn"].mean
        prior_canonical_sample = fwd_trace.nodes["canonical_view"]["value"]
        tb.add_image("prior_samples", torchvision.utils.make_grid(prior_sample), epoch)

        tb.add_image(
            "canonical_prior_samples",
            torchvision.utils.make_grid(prior_canonical_sample),
            epoch,
        )

    pbar = ProgressBar()
    pbar.attach(train_engine)

    @train_engine.on(Events.EPOCH_COMPLETED(every=eval_every))
    def eval(engine):
        eval_engine.run(test_dl, seed=engine.state.epoch)

    train_engine.run(train_dl, max_epochs=50)
