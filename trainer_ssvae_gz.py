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
parser.add_argument('--us_portion', default=0.5, type=float)
parser.add_argument('--subset', default=False, action='store_true')
parser.add_argument('--bar_no_bar', default=False, action='store_true')
args = parser.parse_args()

spec = importlib.util.spec_from_file_location("module.name", args.arch)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
Encoder_y = arch.Encoder_y
Encoder_z = arch.Encoder_z
Decoder = arch.Decoder
use_cuda = not args.no_cuda

if args.bar_no_bar == False:
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    list_of_ans = [a0, a02, a03]
else:
    a01 = "t03_bar_a06_bar_count"
    a02 = "t03_bar_a07_no_bar_count"
    list_of_ans = [a01, a02]

data = Gz2_data(csv_dir=args.csv_file,
                image_dir=args.img_file,
                list_of_interest=list_of_ans,
                crop=args.img_size,
                resize=args.crop_size, one_hot_categorical=True)


encoder_y_args = {'x_size':args.x_size, 'y_size':args.y_size}
encoder_z_args = {'x_size':args.x_size, 'z_size':args.z_size, 'y_size':args.y_size}
decoder_args = {'x_size':args.x_size, 'z_size':args.z_size, 'y_size':args.y_size}
optimizer = Adam({"lr": args.lr})

test_s_loader, test_us_loader, train_s_loader, train_us_loader = return_ss_loader(data, 0.2,
                                                                                  args.us_portion, args.batch_size, subset=args.subset)

ssvae = SSVAE(Encoder_y, Encoder_z, Decoder, args.z_size, args.y_size,
              encoder_y_args, encoder_z_args, decoder_args, use_cuda=use_cuda)


import pyro

#pyro.enable_validation(True)
#data = next(iter(train_s_loader))
#model = ssvae.model
#trace = pyro.poutine.trace(pyro.poutine.enum(model, first_available_dim=-2)).get_trace(data['image'])
#trace.compute_log_prob()
#print(trace.format_shapes())

img_len = args.x_size
#svi = SVI(ssvae.model, ssvae.guide, optimizer, loss=Trace_ELBO())
guide = config_enumerate(ssvae.guide, "parallel", expand=True)
svi = SVI(ssvae.model, guide, optimizer, loss=TraceEnum_ELBO())
print("train and log")
train_log(args.dir_name, ssvae, svi, train_s_loader, train_us_loader, test_s_loader, test_us_loader, img_len,
          args.num_epochs, use_cuda=use_cuda, plot_img_freq=5, checkpoint_freq=50, test_freq=2, testing=args.subset)

