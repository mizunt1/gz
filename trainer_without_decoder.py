from torch.utils.tensorboard import SummaryWriter
import numpy as np
from construct_pose_vae_split import PoseVAE
from galaxy_gen.etn import transforms as T 
from galaxy_gen.etn import transformers, networks

import torch
import torch.nn.functional as f
import torchvision as tv
import os
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as D
import importlib
from classifier_conv import Classifier
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
parser.add_argument('--img_size', default=80, type=int)
parser.add_argument('--lr', default=1.0e-4, type=float)
parser.add_argument('--z_size', default=100, type=int)
parser.add_argument('--crop_size', default=80, type=int)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--subset', default=False, action='store_true')
parser.add_argument('--load_checkpoint', default=None)
parser.add_argument('--bar_no_bar', default=False, action='store_true')
parser.add_argument('--subset_proportion', default=0.5, type=float)

args = parser.parse_args()
spec = importlib.util.spec_from_file_location("module.name", args.arch)
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)
Encoder = arch.Encoder
Decoder = arch.Decoder

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


def evaluate_vae_classifier(vae, vae_loss_fn, classifier, classifier_loss_fn, test_loader, use_cuda=False, transform=False):
    """
    evaluates for all test data
    test data is in batches, all batches in test loader tested
    """
    epoch_loss_vae = 0.
    epoch_loss_classifier = 0.
    total_acc = 0.
    rms = 0.
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
        out, split = vae.encoder(x)
        # combined_z = torch.cat((z_loc, z_scale), 1)
        # combined_z = combined_z.detach()
        y_out = classifier.forward(split)
        classifier_loss = classifier_loss_fn(y_out, y)
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1),y.argmax(dim=1)))
        epoch_loss_vae += vae_loss.item()
        epoch_loss_classifier += classifier_loss.item()
        rms += rms_calc(y_out, y)
    normalizer = len(test_loader.dataset)
    total_epoch_loss_vae = epoch_loss_vae / normalizer
    total_epoch_loss_classifier = epoch_loss_classifier / normalizer
    total_epoch_acc = total_acc / normalizer
    rms_epoch = rms / normalizer
    return total_epoch_loss_classifier, total_epoch_acc, rms_epoch


def train_vae_classifier(vae, vae_optim, vae_loss_fn, classifier, classifier_optim, classifier_loss_fn,
                         train_loader, use_cuda=True, transform=False):
    """
    train vae and classifier for one epoch
    returns loss for one epoch
    in each batch, when the svi takes a step, the optimiser of classifier takes a step
    """
    epoch_loss_classifier = 0.
    total_acc = 0.
    num_steps = 0
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
        out, split = vae.encoder(x)
        # combined_z = torch.cat((z_loc, z_scale), 1)
        y_out = classifier.forward(split)
        classifier_loss = classifier_loss_fn(y_out, y)
        # step through classifier
        total_loss = classifier_loss
        epoch_loss_classifier += classifier_loss.item()
        total_loss.backward()
        classifier_optim.step()
        total_acc += torch.sum(torch.eq(y_out.argmax(dim=1),y.argmax(dim=1)))
        num_steps += 1
    normalizer = len(train_loader.dataset)
    total_epoch_loss_classifier = epoch_loss_classifier / normalizer
    total_acc_norm = total_acc /normalizer
    return total_epoch_loss_classifier, total_acc_norm, num_steps

def rms_calc(probs, target):
    """
    total rms for a single batch
    """
    target = target.cpu().numpy()
    probs = probs.detach().cpu().numpy()
    total_count = np.sum(target, axis=1)
    probs_target = target / total_count[:, None]
    rms =  np.sqrt((probs - probs_target)**2)
    return np.sum(rms)
    
def train_log_vae_classifier(dir_name, vae, vae_optim, vae_loss_fn, classifier, classifier_optim,
                             classifier_loss_fn, train_loader, test_loader, num_epochs, plot_img_freq=20, num_img_plt=10,
                             checkpoint_freq=20, use_cuda=True, test_freq=1, transform=False):
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    writer = SummaryWriter("tb_data_all/" + dir_name)
    total_steps = 0
    if not os.path.exists("checkpoints/" + dir_name):
        os.makedirs("checkpoints/" + dir_name)
    if use_cuda:
        classifier.cuda()
    for epoch in range(num_epochs):
        print("training")
        total_epoch_loss_classifier, total_epoch_acc, num_steps  = train_vae_classifier(
            vae, vae_optim, vae_loss_fn, classifier,
            classifier_optim, classifier_loss_fn, train_loader,
            use_cuda=use_cuda, transform=transform)
        total_steps += num_steps
        print("end train")
        print("[epoch %03d]  average training loss classifier: %.4f" % (epoch, total_epoch_loss_classifier))
        print("[epoch %03d]  average training accuracy: %.4f" % (epoch, total_epoch_acc))
        
        if epoch % test_freq == 0:
            # report test diagnostics
            print("evaluating")
            total_epoch_loss_test_classifier, accuracy, rms = evaluate_vae_classifier(
                vae, vae_loss_fn, classifier, classifier_loss_fn, test_loader,
                use_cuda=use_cuda, transform=transform)
            print("[epoch %03d] average test loss classifier: %.4f" % (epoch, total_epoch_loss_test_classifier))
            print("[epoch %03d] average test accuracy: %.4f" % (epoch, accuracy))
            print("evaluate end")
            writer.add_scalar('Train loss classifier', total_epoch_loss_classifier, total_steps)
            writer.add_scalar('Train accuracy', total_epoch_acc, total_steps)
            writer.add_scalar('Test loss classifier', total_epoch_loss_test_classifier, total_steps)
            writer.add_scalar('Test accuracy', accuracy, total_steps)
            writer.add_scalar('rms normalised', rms, total_steps)
            
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



data = Gz2_data(csv_dir=args.csv_file,
                image_dir=args.img_file,
                list_of_interest=list_of_ans,
                crop=args.img_size,
                resize=args.crop_size)

trans = transformers.TransformerSequence(
    transformers.Translation(networks.EquivariantPosePredictor, 1, 32),
    transformers.Rotation(networks.EquivariantPosePredictor, 1, 32))

encoder_args = {'transformer':trans, 'insize':args.img_size, 'z_dim':args.z_size}
decoder_args = {'z_dim':args.z_size, 'outsize':args.img_size}

test_proportion = 0.1
if args.subset is True:
    train_loader, test_loader = return_subset(data, test_proportion, args.subset_proportion, batch_size=args.batch_size, shuffle=True)
else:
    train_loader, test_loader  = return_data_loader(data, test_proportion, batch_size=args.batch_size, shuffle=True)

print("train and log")



vae = PoseVAE(Encoder, Decoder, args.z_size, encoder_args, decoder_args, use_cuda=use_cuda)
if args.load_checkpoint != None:
    vae.encoder.load_state_dict(torch.load("checkpoints/" + args.load_checkpoint + "/encoder.checkpoint"))
    vae.decoder.load_state_dict(torch.load("checkpoints/" + args.load_checkpoint + "/decoder.checkpoint"))
print("total data:",  len(data))
print("num data points in train_loader:", len(train_loader.dataset))
print("num data points in test_loader:", len(test_loader.dataset))
print("train and log")

vae_optim = Adam(vae.parameters(), lr= args.lr, betas= (0.90, 0.999))

classifier = Classifier(in_dim=vae.encoder.linear_size)

params = list(classifier.parameters()) + list(vae.encoder.parameters())
classifier_optim = Adam(params, args.lr , betas=(0.90, 0.999))


def multinomial_loss(probs, values):
    return torch.sum(-1 *D.Multinomial(1, probs=probs).log_prob(values.float()))

classifier_loss = multinomial_loss

train_log_vae_classifier(args.dir_name, vae, vae_optim, Trace_ELBO().differentiable_loss,
                         classifier, classifier_optim,
                         classifier_loss, train_loader,
                         test_loader, args.num_epochs, use_cuda=use_cuda)
