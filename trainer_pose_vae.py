from construct_pose_vae_split import PoseVAE, train_log
import importlib
from galaxy_gen.etn import transformers, networks
from load_gz_data import Gz2_data, return_data_loader, return_subset
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO
import argparse
from pyro.optim import Adam
parser = argparse.ArgumentParser()
csv = "gz2_data/gz_amended.csv"
img = "gz2_data/"

parser.add_argument('--dir_name', required=True)
parser.add_argument('--arch', required=True)
parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
parser.add_argument('--img_file', metavar='i', type=str, default=img)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--img_size', default=80, type=int)
parser.add_argument('--lr', default=1.0e-3, type=float)
parser.add_argument('--z_size', default=100, type=int)
parser.add_argument('--crop_size', default=80, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--subset', default=False, action='store_true')
parser.add_argument('--subset_proportion', default=0.001, type=float)

args = parser.parse_args()
#import importlib.util
#spec = importlib.util.spec_from_file_location("model","encoder_decoder_32.py")
#model = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(model)
#model.encoder
spec = importlib.util.spec_from_file_location("module.name", args.arch)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
Encoder = arch.Encoder
Decoder = arch.Decoder

use_cuda = not args.no_cuda
a01 = "t01_smooth_or_features_a01_smooth_count"
a02 = "t01_smooth_or_features_a02_features_or_disk_count"
a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
data = Gz2_data(csv_dir=args.csv_file,
                image_dir=args.img_file,
                list_of_interest=[a01,
                                  a02,
                                  a03],
                crop=args.img_size,
                resize=args.crop_size)
trans = transformers.TransformerSequence(
    transformers.Translation(networks.EquivariantPosePredictor, 1, 32),
    transformers.Rotation(networks.EquivariantPosePredictor, 1, 32))
encoder_args = {'insize':args.img_size, 'z_dim':args.z_size, 'transformer':trans}
decoder_args = {'z_dim':args.z_size, 'outsize':args.img_size}
optimizer = Adam({"lr": args.lr})
pvae = PoseVAE(Encoder, Decoder, args.z_size, encoder_args, decoder_args, use_cuda=use_cuda)


test_proportion = 0.2
if args.subset is True:
    train_loader, test_loader = return_subset(data, test_proportion, args.subset_proportion, batch_size=args.batch_size, shuffle=True)
else:
    train_loader, test_loader  = return_data_loader(data, test_proportion, batch_size=args.batch_size, shuffle=True)

svi = SVI(pvae.model, pvae.guide, optimizer, loss=Trace_ELBO())


print("train and log")
train_log(args.dir_name, pvae, svi, train_loader, test_loader, args.num_epochs, use_cuda=use_cuda)
