import argparse
import importlib

from pyro.infer import Trace_ELBO
import pyro.distributions as D
import torch
from torch.optim import Adam

from construct_vae import PoseVAE
from galaxy_gen.etn import transformers, networks
from galaxy_gen.etn import transforms as T
from train_functions import train_ss_epoch, train_fs_epoch, train_log
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



def main():
    parser = argparse.ArgumentParser()
    csv = "gz2_data/gz_amended.csv"
    img = "gz2_data/"

    parser.add_argument('--arch', required=True)
    parser.add_argument('--class_arch', required=True)
    parser.add_argument('--dir_name', required=True)

    parser.add_argument('--bar_no_bar', default=False, action='store_true')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--crop_size', default=80, type=int,
                        help='centre crop image to this size')
    parser.add_argument('--csv_file', metavar='c', type=str,
                        default=csv, help='path to csv')
    parser.add_argument('--img_file', metavar='i', type=str,
                        default=img, help='path to image files')
    parser.add_argument('--img_size', default=80, type=int)
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--lr', default=1.0e-4, type=float)
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--semi-supervised', default=False, action='store_true')
    parser.add_argument('--split_early', default=False, action='store_true')
    parser.add_argument('--subset', default=False, action='store_true',
                        help='use a subset of data for testing')
    parser.add_argument('--subset_proportion', default=100,
                        type=float,
                        help='what proportion of data for subset,\
                        or acutal int amount of data to be used')
    parser.add_argument('--supervised_proportion', default=0.8, type=float,
                        help='what proportion of data is labelled in \
                        semi-sup learning')
    parser.add_argument('--test_proportion', default=0.1,
                        help='what proportion of total data is test')
    parser.add_argument('--z_size', default=100,
                        type=int, help='size of vae latent vector size')

    args = parser.parse_args()
    transform_spec = ['Rotation']
    use_cuda = not args.no_cuda

    ### loading classifier network
    spec = importlib.util.spec_from_file_location("module.name", args.class_arch)
    class_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(class_arch)
    Classifier = class_arch.Classifier

    ### setting up encoder, decoder and vae
    spec = importlib.util.spec_from_file_location("module.name", args.arch)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    Encoder = arch.Encoder
    Decoder = arch.Decoder
    transforms, transformer = get_transformations(transform_spec)
    encoder_args = {'transformer': transformer,
                    'insize': args.img_size, 'z_dim': args.z_size}
    decoder_args = {'z_dim': args.z_size, 'outsize': args.img_size}
    vae = PoseVAE(
        Encoder, Decoder, args.z_size, encoder_args,
        decoder_args, transforms, use_cuda=use_cuda)

    if args.load_checkpoint is not None:
        vae.encoder.load_state_dict(
            torch.load("checkpoints/" + args.load_checkpoint + "/encoder.checkpoint"))
        vae.decoder.load_state_dict(
            torch.load("checkpoints/" + args.load_checkpoint + "/decoder.checkpoint"))

    ### define classifier and optims
    classifier = Classifier(in_dim=vae.encoder.linear_size)
    classifier_params = list(classifier.parameters()) + list(vae.encoder.parameters())
    classifier_optim = Adam(classifier_params, args.lr, betas=(0.90, 0.999))
    vae_optim = Adam(vae.parameters(), lr=args.lr/100, betas=(0.90, 0.999))

    ### count number of parameters in network

    def multinomial_loss(probs, values):
        return torch.sum(
            -1 * D.Multinomial(1, probs=probs).log_prob(values.float()))


    classifier_loss = multinomial_loss

    ### setting up data and dataloaders
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"

    if not args.bar_no_bar:
        a01 = "t01_smooth_or_features_a01_smooth_count"
        a02 = "t01_smooth_or_features_a02_features_or_disk_count"
        a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
        list_of_ans = [a01, a02, a03]
    else:
        a01 = "t03_bar_a06_bar_count"
        a02 = "t03_bar_a07_no_bar_count"
        list_of_ans = [a01, a02]


    data = Gz2_data(csv_dir=args.csv_file,
                    image_dir=args.img_file,
                    list_of_interest=list_of_ans,
                    crop=args.img_size,
                    resize=args.crop_size)

    ### different dataloaders depending on whether its semi-sup or fully sup


    if args.semi_supervised:
        if args.subset:
            test_s_loader, test_us_loader, train_s_loader, train_us_loader = return_ss_loader(
                data, args.test_proportion, args.supervised_proportion, batch_size=args.batch_size,
                shuffle=True, subset_proportion = args.subset_proportion)
        else:
            test_s_loader, test_us_loader, train_s_loader, train_us_loader  = return_ss_loader(
                data, args.test_proportion, args.supervised_proportion, batch_size=args.batch_size, shuffle=True, subset=False)
        test_loader = test_us_loader
        print("semi supervised training")
        print("total data:",  len(data))
        print("num data points in test_s_loader:", len(test_s_loader.dataset))
        print("num data points in test_us_loader:", len(test_us_loader.dataset))
        print("num data points in train_s_loader:", len(train_s_loader.dataset))
        print("num data points in train_us_loader:", len(train_us_loader.dataset))
        print("train and log")
        kwargs = {'train_s_loader': train_s_loader, 'train_us_loader': train_us_loader, 'split_early': args.split_early}
        train_fn = train_ss_epoch

    else:
        if args.subset:
            train_loader, test_loader = return_subset(
                data, args.test_proportion, args.subset_proportion,
                batch_size=args.batch_size, shuffle=True)
        else:
            train_loader, test_loader = return_data_loader(
                data, args.test_proportion,
                batch_size=args.batch_size, shuffle=True)
        print("Fully supervised training")
        print("num data points in test_loader:", len(test_loader.dataset))
        print("num data points in test_loader:", len(train_loader.dataset))

        kwargs = {'train_loader': train_loader, 'split_early': args.split_early}
        train_fn = train_fs_epoch

    train_log(train_fn, vae, vae_optim,
              Trace_ELBO().differentiable_loss,
              classifier, classifier_optim,
              classifier_loss, args.dir_name, args.num_epochs,
              use_cuda, test_loader, kwargs)


if __name__ == '__main__':
    main()
