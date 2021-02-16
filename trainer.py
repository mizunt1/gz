""" trains semi-supervised or fully supervised arch given a yaml config file.
Run with sacred ie python trainer.py with config.yaml --file_storage=my_runs"""

import argparse
import importlib

from pyro.infer import Trace_ELBO
import pyro.distributions as D
from sacred import Experiment
import torch
from torch.optim import Adam
import yaml

from construct_vae import PoseVAE
from galaxy_gen.etn import transformers, networks
from galaxy_gen.etn import transforms as T
from train_functions import train_ss_epoch, train_fs_epoch, train_log, train_vae, train_log_vae
from neural_networks.mike_arch import MikeArch, train_log_mike
from utils.load_gz_data import Gz2_data, return_data_loader, return_subset, return_ss_loader


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


def get_transformations(transform_spec):
    transformer = transformers.TransformerSequence(
        *[TRANSFORMER_STRING_MAP[trans_name](networks.EquivariantPosePredictor, 1, 32) for trans_name in transform_spec]
    )

    transforms = T.TransformSequence(
        *[TRANSFORM_STRING_MAP[trans_name]() for trans_name in transform_spec]
    )
    return transforms, transformer

ex = Experiment()


@ex.automain
def main(dir_name, cuda, num_epochs, semi_supervised,
         train_type, split_early, checkpoints_path,
         subset_proportion, lr_vae, lr_class, supervised_proportion,
         csv_file, img_file, load_checkpoint, arch_classifier,
         arch_vae, test_proportion, z_size, batch_size, bar_no_bar,
         crop_size, img_size, transform_spec):
    ### loading classifier network
    spec = importlib.util.spec_from_file_location(
        "module.name", arch_classifier)
    class_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(class_arch)
    Classifier = class_arch.Classifier
    loss_fn = class_arch.loss
    ### setting up encoder, decoder and vae
    spec = importlib.util.spec_from_file_location("module.name", arch_vae)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    Encoder = arch.Encoder
    Decoder = arch.Decoder
    transforms, transformer = get_transformations(transform_spec)
    encoder_args = {'transformer': transformer,
                    'insize': img_size, 'z_dim': z_size}
    decoder_args = {'z_dim': z_size, 'outsize': img_size}
    vae = PoseVAE(
        Encoder, Decoder, z_size, encoder_args,
        decoder_args, transforms, use_cuda=cuda)

    if load_checkpoint is not None:
        vae.encoder.load_state_dict(
            torch.load(load_checkpoint + "/encoder.checkpoint"))
        vae.decoder.load_state_dict(
            torch.load(load_checkpoint + "/decoder.checkpoint"))

    ### define classifier and optims
    if split_early:
        classifier = Classifier(in_dim=vae.encoder.linear_size)
    else:
        classifier = Classifier(in_dim=z_size)
    classifier_params = list(classifier.parameters()) + list(vae.encoder.parameters())
    classifier_optim = Adam(classifier_params, lr_class, betas=(0.90, 0.999))
    vae_optim = Adam(vae.parameters(), lr=lr_vae, betas=(0.90, 0.999))

    classifier_loss = loss_fn

    ### setting up data and dataloaders
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"

    if not bar_no_bar:
        a01 = "t01_smooth_or_features_a01_smooth_count"
        a02 = "t01_smooth_or_features_a02_features_or_disk_count"
        a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
        list_of_ans = [a01, a02, a03]
    else:
        a01 = "t03_bar_a06_bar_count"
        a02 = "t03_bar_a07_no_bar_count"
        list_of_ans = [a01, a02]


    data = Gz2_data(csv_dir=csv_file,
                    image_dir=img_file,
                    list_of_interest=list_of_ans,
                    crop=img_size,
                    resize=crop_size)

    ### different dataloaders depending on whether its semi-sup or fully sup


    if semi_supervised:
        test_s_loader, test_us_loader, train_s_loader, train_us_loader = return_ss_loader(
            data, test_proportion, supervised_proportion, batch_size=batch_size,
            shuffle=True, subset_unsupervised_proportion = subset_proportion)
        test_loader = test_us_loader
        print("semi supervised training")
        print("total data:",  len(data))
        print("num data points in test_s_loader:", len(test_s_loader.dataset))
        print("num data points in test_us_loader:", len(test_us_loader.dataset))
        print("num data points in train_s_loader:", len(train_s_loader.dataset))
        print("num data points in train_us_loader:", len(train_us_loader.dataset))
        print("train and log")

    else:
        if subset_proportion != None:
            train_loader, test_loader = return_subset(
                data, test_proportion, subset_proportion,
                batch_size=batch_size, shuffle=True)
        else:
            train_loader, test_loader = return_data_loader(
                data, test_proportion,
                batch_size=batch_size, shuffle=True)
        print("no splitting of supervised and unsupervised data")
        print("num data points in test_loader:", len(test_loader.dataset))
        print("num data points in train_loader:", len(train_loader.dataset))

    if train_type == "standard_semi-supervised":
        train_fn = train_ss_epoch
        kwargs = {'train_s_loader': train_s_loader,
                  'train_us_loader': train_us_loader}
        train_log(train_fn, vae, vae_optim,
                  Trace_ELBO().differentiable_loss,
                  classifier, classifier_optim,
                  classifier_loss, dir_name, checkpoints_path, num_epochs,
                  cuda, test_loader, split_early, kwargs)

    elif train_type == "standard_fully_supervised":
        train_fn = train_fs_epoch
        kwargs = {'train_loader': train_loader}
        train_log(train_fn, vae, vae_optim,
                  Trace_ELBO().differentiable_loss,
                  classifier, classifier_optim,
                  classifier_loss, dir_name, checkpoints_path, num_epochs,
                  cuda, test_loader, split_early, kwargs)

    elif train_type == "vae_only":
        train_fn = train_vae
        train_log_vae(train_fn, vae, vae_optim,
                      Trace_ELBO().differentiable_loss, transform_spec,
                      train_loader, test_loader, cuda, split_early,
                      dir_name, checkpoints_path, num_epochs, checkpoint_freq=5, num_img_plt=9,  test_freq=1, plt_img_freq=1)

    elif train_type == "mike":
        classifier = MikeArch
        train_log_mike(dir_name, classifier, classifier_optim,
                       train_loader, test_freq=1,
                       num_epochs=num_epochs, use_cuda=cuda)

    elif train_type == "bayes_semi_supervised":
        train_fn = train_ss_bayes
        guide = AutoDiagonalNormal(model)
        classifier_optim = pyro.optim.Adam({"lr": 0.03})
        kwargs = {'train_s_loader': train_s_loader,
                  'train_us_loader': train_us_loader, 'guide':guide}

        train_log(train_ss_bayes, vae, vae_optim,
                  Trace_ELBO().differentiable_loss,
                  classifier, classifier_optim,
                  Trace_ELBO.differentiable_loss, dir_name, num_epochs,
                  cuda, test_loader, split_early, kwargs, bayesian=True)
