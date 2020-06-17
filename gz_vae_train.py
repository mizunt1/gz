from gz_vae_conv import VAE, train, evaluate
from torch.utils.tensorboard import SummaryWriter
from load_gz_data import Gz2_data, return_data_loader
from simple_classifier import Classifier
import torch
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import argparse
parser = argparse.ArgumentParser()

csv = "gz2_mini/gz2_4.csv"
img = "gz2_mini/"
parser.add_argument('--writer', required=True)
parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
parser.add_argument('--img_file', metavar='i', type=str, default=img)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--num_epochs', type=int, default=3)

writer = SummaryWriter("tb_data_all/" + args.writer)
use_cuda = not args.no_cuda
args = parser.parse_args()

a01 = "t01_smooth_or_features_a01_smooth_count"
a02 = "t01_smooth_or_features_a02_features_or_disk_count"
a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
data = Gz2_data(csv_dir=args.csv_file,
                image_dir=args.img_file,
                list_of_interest=[a01,
                                  a02,
                                  a03])

vae = VAE(use_cuda=use_cuda)

optimizer = Adam({"lr": 1.0e-3})

svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())


batch_size = 1
test_proportion = 0.5
train_loader, test_loader = return_data_loader(data, test_proportion, batch_size)

test_freq = 1
# training VAE
plot_img_freq = 1

for epoch in range(args.num_epochs):
    print("training")
    total_epoch_loss_train = train(svi, train_loader, use_cuda=use_cuda)
    print("end train")
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
    if epoch % test_freq == 0:
        # report test diagnostics
        print("evaluating")
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
        print("evaluate end")
        writer.add_scalar('Train loss', total_epoch_loss_train, epoch)
        writer.add_scalar('Test loss', total_epoch_loss_test, epoch)

    if epoch % plot_img_freq == 0:
        one_image = next(iter(train_loader))['image'][0]
        print("one_image", one_image.shape)
        images_out = vae.sample_img(one_image, use_cuda=use_cuda)
        img_grid = torchvision.utils.make_grid(images_out)
        writer.add_image('images from epoch'+ int(epoch), img_grid)
    writer.close()
