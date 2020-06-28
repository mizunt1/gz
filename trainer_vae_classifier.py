from torch.utils.tensorboard import SummaryWriter
from construct_vae import VAE, evaluate, train_log_vae
import torch
import torchvision as tv
import os
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
import importlib
from classifier_gz import Classifier
from load_gz_data import Gz2_data, return_data_loader, return_subset
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO
import argparse
from torch.optim import Adam
parser = argparse.ArgumentParser()
csv = "gz2_data/gz_amended.csv"
img = "gz2_data/"

parser.add_argument('--dir_name', required=True)
parser.add_argument('--arch', required=True)
parser.add_argument('--csv_file', metavar='c', type=str, default=csv)
parser.add_argument('--img_file', metavar='i', type=str, default=img)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--img_size', default=56, type=int)
parser.add_argument('--lr', default=1.0e-3, type=float)
parser.add_argument('--z_size', default=100, type=int)
parser.add_argument('--crop_size', default=56, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--subset', default=False, action='store_true')
parser.add_argument('--load_checkpoint', default=None)
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

encoder_args = {'insize':args.img_size, 'z_dim':args.z_size}
decoder_args = {'z_dim':args.z_size, 'outsize':args.img_size}




test_proportion = 0.2
if args.subset is True:
    train_loader, test_loader = return_subset(data, test_proportion, 128, batch_size=args.batch_size, shuffle=True)
else:
    train_loader, test_loader  = return_data_loader(data, test_proportion, batch_size=args.batch_size, shuffle=True)




print("train and log")


def evaluate_vae_classifier(vae, vae_loss_fn, classifier, classifier_loss_fn, test_loader, use_cuda=False, transform=False):
    """
    evaluates for all test data
    test data is in batches, all batches in test loader tested
    """
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    for data in test_loader:
        x = data['image']
        y = data['data']
        if transform != False:
            x = transform(x)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # step of elbo for vae
        vae_loss = vae_loss_fn(vae.model, vae.guide, x)
        z_loc, z_scale = vae.encoder(x)
        combined_z = torch.cat((z_loc, z_scale), 1)
        combined_z = combined_z.detach()
        y_out = classifier.forward(combined_z)
        _, y_one_hot = y.max(1)
        classifier_loss = classifier_loss_fn(y_out, y_one_hot)
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1),y.argmax(dim=1)))
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()

    normalizer = len(test_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / normalizer
    total_epoch_loss_classifier = epoch_loss_classifier / normalizer
    total_epoch_acc = total_acc / normalizer
    return total_epoch_loss_vae, total_epoch_loss_classifier, total_epoch_acc 


def train_vae_classifier(vae, vae_optim, vae_loss_fn, classifier, classifier_optim, classifier_loss_fn,
                         train_loader, use_cuda=True, transform=False):
    """
    train vae and classifier for one epoch
    returns loss for one epoch
    in each batch, when the svi takes a step, the optimiser of classifier takes a step
    """
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    for data in train_loader:
        x = data['image']
        y = data['data']
        if transform != False:
            x = transform(x)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # step of elbo for vae
        classifier_optim.zero_grad()
        vae_optim.zero_grad()
        vae_loss = vae_loss_fn(vae.model, vae.guide, x)
        z_loc, z_scale = vae.encoder(x)
        combined_z = torch.cat((z_loc, z_scale), 1)
        combined_z = combined_z.detach()
        y_out = classifier.forward(combined_z)
        _, y_one_hot = y.max(1)
        classifier_loss = classifier_loss_fn(y_out, y_one_hot)
        # step through classifier
        total_loss = vae_loss + classifier_loss
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        total_loss.backward()
        vae_optim.step()
        classifier_optim.step()
    normalizer = len(train_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / normalizer
    total_epoch_loss_classifier = epoch_loss_classifier / normalizer
    return total_epoch_loss_vae, total_epoch_loss_classifier


vae = VAE(Encoder, Decoder, args.z_size, encoder_args, decoder_args, use_cuda=use_cuda)
if args.load_checkpoint != None:
    vae.encoder.load_state_dict(torch.load("checkpoints/" + args.load_checkpoint + "/encoder.checkpoint"))
    vae.decoder.load_state_dict(torch.load("checkpoints/" + args.load_checkpoint + "/decoder.checkpoint"))
vae_optim = Adam(vae.parameters(), lr= args.lr, betas= (0.90, 0.999))

#vae_optim = Adam({"lr": 0.001})
classifier = Classifier(in_dim=args.z_size*2)

classifier_optim = Adam(classifier.parameters(),lr=1e-4, betas=(0.90, 0.999))
# or optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)?
classifier_loss = nn.CrossEntropyLoss()


def train_log_vae_classifier(dir_name, vae, vae_optim, vae_loss_fn, classifier, classifier_optim,
                             classifier_loss_fn, train_loader, test_loader, num_epochs, plot_img_freq=1, num_img_plt=40,
                             checkpoint_freq=20, use_cuda=True, test_freq=1, transform=False):
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    writer = SummaryWriter("tb_data_all/" + dir_name)
    if not os.path.exists("checkpoints/" + dir_name):
        os.makedirs("checkpoints/" + dir_name)
    if use_cuda:
        classifier.cuda()
    for epoch in range(num_epochs):
        print("training")
        total_epoch_loss_vae, total_epoch_loss_classifier = train_vae_classifier(vae, vae_optim, vae_loss_fn, classifier,
                                                                                 classifier_optim, classifier_loss_fn, train_loader,
                                                                                 use_cuda=use_cuda, transform=transform)
        print("end train")
        print("[epoch %03d]  average training loss vae: %.4f" % (epoch, total_epoch_loss_vae))
        print("[epoch %03d]  average training loss classifier: %.4f" % (epoch, total_epoch_loss_classifier))
        if epoch % test_freq == 0:
            # report test diagnostics
            print("evaluating")
            total_epoch_loss_test_vae, total_epoch_loss_test_classifier, accuracy = evaluate_vae_classifier(
                vae, vae_loss_fn, classifier, classifier_loss_fn, test_loader,
                use_cuda=use_cuda, transform=transform)
            print("[epoch %03d] average test loss vae: %.4f" % (epoch, total_epoch_loss_test_vae))
            print("[epoch %03d] average test loss classifier: %.4f" % (epoch, total_epoch_loss_test_classifier))
            print("[epoch %03d] average accuracy: %.4f" % (epoch, accuracy))
            print("evaluate end")
            writer.add_scalar('Train loss vae', total_epoch_loss_vae, epoch)
            writer.add_scalar('Train loss classifier', total_epoch_loss_classifier, epoch)
            writer.add_scalar('Test loss vae', total_epoch_loss_test_vae, epoch)
            writer.add_scalar('Test loss classifier', total_epoch_loss_test_classifier, epoch)
            writer.add_scalar('Test accuracy', accuracy, epoch)
            print(epoch)
        if epoch % plot_img_freq == 0:
            image_in = next(iter(train_loader))['image'][0:num_img_plt]
            images_out = vae.sample_img(image_in, use_cuda=use_cuda)
            img_grid_in = tv.utils.make_grid(image_in)
            img_grid = tv.utils.make_grid(images_out)
            writer.add_image('images in, from epoch' + str(epoch), img_grid_in)
            writer.add_image(str(num_params) + ' images out, from epoch'+ str(epoch), img_grid)

        if epoch % checkpoint_freq == 0:

            torch.save(vae.encoder.state_dict(), "checkpoints/" + dir_name + "/encoder.checkpoint")
            torch.save(vae.decoder.state_dict(),  "checkpoints/" + dir_name +  "/decoder.checkpoint")
            torch.save(classifier.state_dict(),  "checkpoints/" + dir_name +  "/classfier.checkpoint")
            
        writer.close()


train_log_vae_classifier(args.dir_name, vae, vae_optim, Trace_ELBO().differentiable_loss,
                         classifier, classifier_optim,
                         classifier_loss, train_loader,
                         test_loader, args.num_epochs, use_cuda=use_cuda)
