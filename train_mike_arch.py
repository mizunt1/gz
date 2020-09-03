from mike_arch import Mike, train_log
from torch.optim import Adam
from load_gz_data import Gz2_data, return_data_loader, return_subset
import argparse
parser = argparse.ArgumentParser()
csv = "gz2_data/gz_amended.csv"
img = "gz2_data/"

parser.add_argument('--dir_name', required=True)
parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
parser.add_argument('--img_file', metavar='i', type=str, default=img)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--img_size', default=128, type=int)
parser.add_argument('--crop_size', default=128, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--subset', default=False, action='store_true')
parser.add_argument('--subset_proportion', default=0.05, type=float)

parser.add_argument('--bar_no_bar', default=False, action='store_true')

args = parser.parse_args()


use_cuda = not args.no_cuda
a01 = "t01_smooth_or_features_a01_smooth_count"
a02 = "t01_smooth_or_features_a02_features_or_disk_count"
a03 = "t01_smooth_or_features_a03_star_or_artifact_count"

if args.bar_no_bar == False:
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


test_proportion = 0.1
if args.subset is True:
    train_loader, test_loader = return_subset(data, test_proportion, args.subset_proportion, batch_size=args.batch_size, shuffle=True)
else:
    train_loader, test_loader  = return_data_loader(data, test_proportion, batch_size=args.batch_size, shuffle=True)

print("train and log")
print("train len", len(train_loader.dataset))
classifier = Mike()
classifier_optim = Adam(classifier.parameters(), lr=args.lr, betas=(0.90, 0.999))
train_log(args.dir_name, classifier, classifier_optim, train_loader,
          test_loader, test_freq=1, num_epochs=args.num_epochs, use_cuda=use_cuda)
