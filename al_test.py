import math
import numpy as np
import importlib
from tqdm.auto import tqdm
import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import datasets
from torchvision import transforms as tvt
from kornia.augmentation import RandomRotation
from construct_vae import PoseVAE
from sacred import Experiment
from pyro.infer import Trace_ELBO
from utils.load_gz_data import (
    Gz2_data,
)  # , return_data_loader, return_subset, return_ss_loader


from torch import distributions as D

from batchbald_redux import (
    active_learning,
    batchbald,
    consistent_mc_dropout,
    joint_entropy,
    repeated_mnist,
)

from trainer import get_transformations


ex = Experiment()


@ex.config
def config():
    dir_name = "sample_100zs_semi_sup_1200_"
    cuda = True
    num_epochs = 200
    semi_supervised = True
    split_early = False
    subset_proportion = None
    csv_file = "/scratch-ssd/oatml/data/gz2/gz2_classifications_and_subjects.csv"
    img_file = "/scratch-ssd/oatml/data/gz2"
    load_checkpoint = False
    lr = 1.0e-4
    supervised_proportion = 1200
    arch_classifier = "neural_networks/classifier_fc.py"
    arch_vae = "neural_networks/encoder_decoder.py"
    test_proportion = 0.1
    z_size = 100
    bar_no_bar = False
    batch_size = 10
    crop_size = 128
    
    split_early = False
    use_pose_encoder = True # should also add an option to use a pretrained model
    pretrain_epochs = 50
    transform_spec = ["Rotation"]
    dataset = "FashionMNIST"
    img_size = 32 if dataset == "FashionMNIST" else 128

class BayesianCNN(consistent_mc_dropout.BayesianModule):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc1 = nn.Linear(1600, 128)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: torch.Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))

        input = torch.flatten(input, -3, -1)

        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input


class MikeCNN(
    consistent_mc_dropout.BayesianModule
):  # the exact architecture used in Walmsley, Smith et. al.
    def __init__(self, nc):
        super().__init__()

        def make_conv(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=3, padding=1),
                nn.ReLU(),
                consistent_mc_dropout.ConsistentMCDropout2d(),
            )

        self.body = nn.Sequential(
            make_conv(1, 32),  # 128
            make_conv(32, 32),
            nn.MaxPool2d(2),  # 64
            make_conv(32, 32),
            make_conv(32, 32),
            nn.MaxPool2d(2),  # 32
            make_conv(32, 16),
            make_conv(16, 16),
            nn.Flatten(-3, -1),
            nn.Linear(16 * 32 * 32, 128),
            consistent_mc_dropout.ConsistentMCDropout(),
            nn.Linear(128, nc),
            nn.LogSoftmax(-1)
        )

    def mc_forward_impl(self, x):
        return self.body(x)


# TODO: enable dataset switching?
# TODO: add support for mikes cnn
# TODO: add support for actual galaxy zoo.


@ex.capture
def get_datasets(dataset):
    datasets = {"FashionMNIST": get_fashionMNIST, "gz": get_gz_data}
    f = datasets.get(dataset)
    if f is not None:
        return f()
    else:
        raise NotImplementedError(
            f"Unknown dataset {dataset}, avaliable options are {set(datasets.keys())}"
        )


def get_fashionMNIST():
    train_dataset = datasets.FashionMNIST(
        "/scratch-ssd/oatml/data/", download=True, train=True, transform=tvt.ToTensor()
    )
    test_dataset = datasets.FashionMNIST(
        "/scratch-ssd/oatml/data", download=True, train=False, transform=tvt.ToTensor()
    )

    # want to create a rotated dataset but not by resampling rotations randomly, as this kind of screws up the active learning argument.
    transform = RandomRotation(180)
    td = train_dataset.data[:, None, ...].float() / 255
    vd = test_dataset.data[:, None, ...].float() / 255

    tt = train_dataset.targets
    vt = test_dataset.targets

    # apply transforms and pad
    td = transform(td)
    vd = transform(vd)

    td = F.pad(td, (2, 2, 2, 2))
    vd = F.pad(vd, (2, 2, 2, 2))
    tds = torch.utils.data.TensorDataset(td, tt)
    vds = torch.utils.data.TensorDataset(vd, vt)

    tds.data = tds.tensors[0]
    tds.targets = tds.tensors[1]

    vds.data = vds.tensors[0]
    vds.targets = vds.tensors[1]
    return tds, vds


@ex.capture
def get_model(
    arch_classifier, arch_vae, transform_spec, split_early, z_size, img_size, cuda
):
    ### loading classifier network
    spec = importlib.util.spec_from_file_location("module.name", arch_classifier)
    class_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(class_arch)
    Classifier = class_arch.Classifier

    ### setting up encoder, decoder and vae
    spec = importlib.util.spec_from_file_location("module.name", arch_vae)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    Encoder = arch.Encoder
    Decoder = arch.Decoder
    transforms, transformer = get_transformations(transform_spec)
    encoder_args = {"transformer": transformer, "insize": img_size, "z_dim": z_size}
    decoder_args = {"z_dim": z_size, "outsize": img_size}
    vae = PoseVAE(
        Encoder, Decoder, z_size, encoder_args, decoder_args, transforms, use_cuda=cuda
    )

    if split_early:
        classifier = Classifier(in_dim=vae.encoder.linear_size)
    else:
        classifier = Classifier(in_dim=z_size)

    return vae, classifier


@ex.capture
def get_gz_data(csv_file, img_file, bar_no_bar, img_size, crop_size, test_proportion):
    ans = {
        False: [
            "t01_smooth_or_features_a01_smooth_count",
            "t01_smooth_or_features_a02_features_or_disk_count",
            "t01_smooth_or_features_a03_star_or_artifact_count",
        ],
        True: [
            "t03_bar_a06_bar_count",
            "t03_bar_a07_no_bar_count",
        ],
    }

    data = Gz2_data(
        csv_dir=csv_file,
        image_dir=img_file,
        list_of_interest=ans[bar_no_bar],
        crop=img_size,
        resize=crop_size,
    )

    len_data = len(data)
    num_tests = int(len_data * test_proportion)
    test_indices = list(i for i in range(0, num_tests))
    train_indices = list(i for i in range(num_tests, len_data))
    test_set = torch.utils.data.Subset(data, test_indices)
    train_set = torch.utils.data.Subset(data, train_indices)
    return train_set, test_set

@ex.capture
def get_classification_model(dataset, bar_no_bar):
    if dataset == "FashionMNIST":
        return BayesianCNN()
    else:
        return MikeCNN(2 if bar_no_bar else 3)


def multinomial_loss(logits, observations, reduction="mean"):
    """
    the nll of a multinomial distirbution parameterised by logits.
    """
    p = D.Multinomial(logits=logits)
    if reduction=="mean":
        return -p.log_prob(observations).mean()
    elif reduction == "sum":
        return -p.log_prob(observations).sum()
    raise NotImplementedError(f"Unknown reduction {reduction}")

@ex.capture
def get_classification_loss(dataset):
    if dataset == "gz":
        return multinomial_loss
    else:
        return F.nll_loss



@ex.automain
def main(use_pose_encoder, pretrain_epochs, dataset, bar_no_bar):

    train_dataset, test_dataset = get_datasets()


    # sanity for initial training
    train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(10000))
    num_initial_samples = 20
    num_classes = 10

    # TODO going to have to change this for galaxy zoo in all likelihood but it will do for now
    if dataset == "gz":
        # in this case getting initial samples is slightly complicated by the fact that galaxy zoo does not have
        # strict labels. 
        # Compromise by balancing as though the arg-max of the votes is a label, which should be a pretty good proxy.
        labels = [x['data'].argmax() for x in train_dataset]
        num_classes = 2 if bar_no_bar else 3
        initial_samples = active_learning.get_balanced_sample_indices(
            labels,
            num_classes=num_classes,
            n_per_digit=num_initial_samples / num_classes
        )
    else:
        initial_samples = active_learning.get_balanced_sample_indices(
            repeated_mnist.get_targets(train_dataset),
            num_classes=num_classes,
            n_per_digit=num_initial_samples / num_classes,
        )

    max_training_samples = 150
    acquisition_batch_size = 5
    num_inference_samples = 25
    num_test_inference_samples = 5
    num_samples = 100000

    test_batch_size = 128
    batch_size = 64
    scoring_batch_size = 32
    training_iterations = 4096 * 6

    use_cuda = torch.cuda.is_available()

    print(f"use_cuda: {use_cuda}")

    device = "cuda" if use_cuda else "cpu"

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs
    )

    active_learning_data = active_learning.ActiveLearningData(train_dataset)

    # Split off the initial samples first.
    active_learning_data.acquire(initial_samples)

    # THIS REMOVES MOST OF THE POOL DATA. UNCOMMENT THIS TO TAKE ALL UNLABELLED DATA INTO ACCOUNT!
    # active_learning_data.extract_dataset_from_pool(40000)

    train_loader = torch.utils.data.DataLoader(
        active_learning_data.training_dataset,
        sampler=active_learning.RandomFixedLengthSampler(
            active_learning_data.training_dataset, training_iterations
        ),
        batch_size=batch_size,
        **kwargs,
    )

    pool_loader = torch.utils.data.DataLoader(
        active_learning_data.pool_dataset,
        batch_size=scoring_batch_size,
        shuffle=False,
        **kwargs,
    )

    # Run experiment
    test_accs = []
    test_loss = []
    added_indices = []

    pbar = tqdm(
        initial=len(active_learning_data.training_dataset),
        total=max_training_samples,
        desc="Training Set Size",
    )

    vae, clf = get_model()
    vae_opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    vae_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    print("starting pretraining")
    if use_pose_encoder:
        # need to pretrain the pose encoder.
        for e in range(pretrain_epochs):
            lb = []
            for batch in vae_loader:
                if isinstance(batch, dict):
                    x = batch['image']
                else:
                    x = batch[0]
                x = x.cuda()
                vae_opt.zero_grad()

                loss = (
                    Trace_ELBO().differentiable_loss(vae.model, vae.guide, x)
                    / batch_size
                )
                loss.backward()
                vae_opt.step()
                lb.append(loss.item())
            print("pretrain epoch", e, "average loss", np.mean(lb))
    print("done pretraining")

    # todo want a switch on BayesianCNN / PoseVAE here.

    while True:
        model = get_classification_model()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = get_classification_loss()
        model.cuda()
        model.train()

        # Train
        for batch in tqdm(train_loader, desc="Training", leave=False):

            if isinstance(batch, dict):
                data = batch['image']
                target = batch['data']

            else:
                data, target = batch

            data = data.to(device=device)
            target = target.to(device=device)
            if use_pose_encoder: # could replace this with a configurable function
                enc_output, _ = vae.encoder(data)  # TODO visualise
                data = enc_output["view"]  # learned transform of the data.
            optimizer.zero_grad()

            prediction = model(data, 1).squeeze(1)
            # loss = F.nll_loss(prediction, target)
            loss = loss_fn(prediction, target)

            loss.backward()
            optimizer.step()

        # Test
        loss = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", leave=False):
                if isinstance(batch, dict):
                    data = batch['image']
                    target = batch['data']

                else:
                    data, target = batch

                data = data.to(device=device)
                target = target.to(device=device)
                if use_pose_encoder:
                    enc_output, _ = vae.encoder(data)  # TODO visualise
                    data = enc_output["view"]  # learned transform of the data.

                prediction = torch.logsumexp(
                    model(data, num_test_inference_samples), dim=1
                ) - math.log(num_test_inference_samples)
                # loss += F.nll_loss(prediction, target, reduction="sum")
                loss += loss_fn(prediction, target, reduction="sum")

                prediction = prediction.max(1)[1]
                if len(target.shape) > 1:
                    target = target.argmax(-1)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        loss /= len(test_loader.dataset)
        test_loss.append(loss)

        percentage_correct = 100.0 * correct / len(test_loader.dataset)
        test_accs.append(percentage_correct)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                loss, correct, len(test_loader.dataset), percentage_correct
            )
        )

        if len(active_learning_data.training_dataset) >= max_training_samples:
            break

        # Acquire pool predictions
        N = len(active_learning_data.pool_dataset)
        logits_N_K_C = torch.empty(
            (N, num_inference_samples, num_classes),
            dtype=torch.double,
            pin_memory=use_cuda,
        )

        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(
                tqdm(pool_loader, desc="Evaluating Acquisition Set", leave=False)
            ):

                if isinstance(batch, dict):
                    data = batch['image']
                else:
                    data = batch[0]

                data = data.to(device=device)
                if use_pose_encoder:
                    enc_output, _ = vae.encoder(data)  # TODO visualise
                    data = enc_output["view"]  # learned transform of the data.

                lower = i * pool_loader.batch_size
                upper = min(lower + pool_loader.batch_size, N)
                logits_N_K_C[lower:upper].copy_(
                    model(data, num_inference_samples).double(), non_blocking=True
                )

        with torch.no_grad():
            candidate_batch = batchbald.get_bald_batch(
                logits_N_K_C,
                acquisition_batch_size,
                # num_samples,#
                dtype=torch.double,
                device=device,
            )

        # targets = repeated_mnist.get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(
            candidate_batch.indices
        )

        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        # print("Labels: ", targets[candidate_batch.indices])

        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(dataset_indices)
        pbar.update(len(dataset_indices))
