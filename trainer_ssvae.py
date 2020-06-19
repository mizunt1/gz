from construct_ssvae import SSVAE, train_log

from ss_encoders_decoders import Encoder_y, Encoder_z, Decoder
from construct_ssvae import SSVAE, train_log
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO
from load_gz_data import Gz2_data, return_data_loader
from torch.utils.data import DataLoader
from load_mnist import return_ss_loader, transform
from pyro.infer import SVI, Trace_ELBO
import argparse
from pyro.optim import Adam
parser = argparse.ArgumentParser()
csv = "gz2_data/gz2_20.csv"
img = "gz2_data/"

parser.add_argument('--dir_name', required=True)
parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
parser.add_argument('--img_file', metavar='i', type=str, default=img)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--img_size', default=56, type=int)
parser.add_argument('--lr', default=1.0e-3, type=float)
parser.add_argument('--input_size_y', default=784, type=int)
# input to encoder_y. i.e. for fully connected, flattened img size
parser.add_argument('--output_size_y', default=10, type=int)
# output from encoder y. i.e. number of categories
parser.add_argument('--input_size_z', default=794, type=int)
# inputsize to encoder_z. Y and X are inputted after cat usually
parser.add_argument('--output_size_z', default=100, type=int)
# output of encoder z, i.e. z dims
parser.add_argument('--input_size_de', default=110, type=int)
# input to the decoder. Y and Z catted to output X
# output from decoder, same as image 

parser.add_argument('--crop_size', default=56, type=int)
parser.add_argument('--batch_size', default=100, type=int)

args = parser.parse_args()
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


encoder_y_args = {'input_size':args.input_size_y, 'output_size':args.output_size_y}
encoder_z_args = {'input_size':args.input_size_z, 'output_size':args.output_size_z}
decoder_args =  {'input_size':args.input_size_z, 'output_size':args.output_size_z}
decoder_args = {'input_size':args.input_size_de, 'output_size':args.input_size_y}
optimizer = Adam({"lr": 1.0e-4})
ssvae = SSVAE(Encoder_y, Encoder_z, Decoder, args.output_size_z, args.output_size_y,
              encoder_y_args, encoder_z_args, decoder_args, use_cuda=use_cuda)
us_portion = 0.2
batch_size = 100
test_s_loader, test_us_loader, train_s_loader, train_us_loader = return_ss_loader(us_portion, args.batch_size)
guide = config_enumerate(ssvae.guide, "parallel", expand=True)
svi = SVI(ssvae.model, guide, optimizer, loss=TraceEnum_ELBO())
print("train and log")
train_log(args.dir_name, ssvae, svi, train_s_loader, train_us_loader, test_s_loader, test_us_loader,
          10, use_cuda=use_cuda)

