from construct_ssvae import SSVAE, train_log
import torchvision as tv
from ss_encoders_decoders import Encoder_y, Encoder_z, Decoder
from construct_ssvae import SSVAE, train_log
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO
from load_gz_data import Gz2_data, return_data_loader
from torch.utils.data import DataLoader
from load_gz_data import return_ss_loader
from pyro.infer import SVI, Trace_ELBO
import argparse
from pyro.optim import Adam
import importlib.util
parser = argparse.ArgumentParser()
csv = "gz2_data/gz_amended.csv"
img = "gz2_data/"

parser.add_argument('--dir_name', required=True)
parser.add_argument('--arch', required=True)
parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
parser.add_argument('--img_file', metavar='i', type=str, default=img)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', default=1.0e-3, type=float)
parser.add_argument('--x_size', default=80, type=int)
# input to encoder_y. i.e. for fully connected, flattened img size
parser.add_argument('--y_size', default=3, type=int)
# output from encoder y. i.e. number of categories
# inputsize to encoder_z. Y and X are inputted after cat usually
parser.add_argument('--z_size', default=100, type=int)
# output of encoder z, i.e. z dims

# output from decoder, same as image 
parser.add_argument('--img_size', default=80, type=int)
parser.add_argument('--crop_size', default=80, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--us_portion', default=0.98, type=float)

args = parser.parse_args()

spec = importlib.util.spec_from_file_location("module.name", args.arch)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
Encoder_y = arch.Encoder_y
Encoder_z = arch.Encoder_z
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
                resize=args.crop_size, one_hot_categorical=True)


encoder_y_args = {'x_size':args.x_size, 'y_size':args.y_size}
encoder_z_args = {'x_size':args.x_size, 'z_size':args.z_size, 'y_size':args.y_size}
decoder_args = {'x_size':args.x_size, 'z_size':args.z_size, 'y_size':args.y_size}
optimizer = Adam({"lr": args.lr})

test_s_loader, test_us_loader, train_s_loader, train_us_loader = return_ss_loader(data, 0.2,
                                                                                  args.us_portion, args.batch_size)

ssvae = SSVAE(Encoder_y, Encoder_z, Decoder, args.z_size, args.y_size,
              encoder_y_args, encoder_z_args, decoder_args, use_cuda=use_cuda)


#import pyro
#pyro.enable_validation(True)

batch_size = 100
img_len = args.x_size
svi = SVI(ssvae.model, ssvae.guide, optimizer, loss=Trace_ELBO())
#guide = config_enumerate(ssvae.guide, "parallel", expand=True)
#svi = SVI(ssvae.model, guide, optimizer, loss=TraceEnum_ELBO())
print("train and log")
train_log(args.dir_name, ssvae, svi, train_s_loader, train_us_loader, test_s_loader, test_us_loader, img_len,
          args.num_epochs, use_cuda=use_cuda, plot_img_freq=5, checkpoint_freq=50, test_freq=2)

