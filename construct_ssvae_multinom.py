import torch.nn as nn
import numpy as np 
import torch
import pyro
import pyro.distributions as dist
import os
from load_mnist import setup_data_loaders, return_data_loader, return_ss_loader
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import utils
import torchvision as tv
from pyro.infer import config_enumerate

class SSVAE(nn.Module):
    def __init__(self, encoder_y, encoder_z, decoder, z_dim, y_dim, encoder_y_args, encoder_z_args, decoder_args, use_cuda=False):
        super().__init__()
        self.encoder_y = encoder_y(**encoder_y_args)
        self.encoder_z = encoder_z(**encoder_z_args)
        self.decoder = decoder(**decoder_args)
        self.z_dim = z_dim
        self.y_dim = y_dim
        if use_cuda:
            self.cuda()
    @config_enumerate
    def model(self, xs, y=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)
        batch_size = xs.size(0)

            # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # sample the handwriting style from the constant prior distribution
            prior_loc = xs.new_zeros([batch_size, self.z_dim])
            prior_scale = xs.new_ones([batch_size, self.z_dim])
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))
            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = xs.new_ones([batch_size, self.y_dim]) / (1.0 * self.y_dim)
            # vector of probabilities for each class, i.e. output_size
            # its a uniform prior
            # making labels one hot for onehotcat
            ys = pyro.sample("y", dist.Multinomial(logits=alpha_prior), obs=y)
            
            # one of the categories will be sampled, according to the distribution specified by alpha prior    
            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            loc = self.decoder.forward(zs, ys)
            # decoder networks takes a category, and a latent variable and outputs an observation x.

            pyro.sample("x", dist.Bernoulli(loc).to_event(3), obs=xs)

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            batch_size = xs.size(0)

            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                # if there is an unlabbeld datapoint, we take the values for x the observations,
                # and we output an alpha which parameterises the classifier.
                alpha = self.encoder_y.forward(xs)
                # then we sample a classification using this parameterisation of the classifier.
                # the classifier is also like a generative model, where given the latents alpha, we 
                # output an observation y

                # and the latents alpha are given by an encoder

                ys = pyro.sample("y", dist.Multinomial(logits=alpha))
                # if the labels y is known, then we dont have to sample from the above,
                # we just feed the actual y in to the encoder that takes x and y.
        
                # sample (and score) the latent handwriting-style with the variational
                # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            # change ys to one hot should do this somewhere else TODO

            loc, scale = self.encoder_z.forward(xs, ys)
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def sample_img(self, x, y, img_width, use_cuda=False):
        # encode image x
        if use_cuda == True:
            x = x.cuda()
            y = y.cuda()

        img_dim = x.shape[-1]
        z_loc, z_scale = self.encoder_z(x,y)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z,y)
        return loc_img.reshape([-1, 1, img_width, img_width])

    def test_acc(self, test_loader, use_cuda=True):
        # tests one batch
        data = next(iter(test_loader))
        image_in = data['image']
        labels = data['data']
        if use_cuda == True:
            image_in = image_in.cuda()
            labels = labels.cuda()
        out = self.encoder_y.forward(image_in)
        _, max_ind = torch.max(out, 1)
        _, max_ind_label = torch.max(labels, 1)
        acc_per_batch = torch.sum(max_ind_label == max_ind)
        return acc_per_batch.item()/ out.shape[0]

    def rms_calc(self, test_loader, use_cuda=True):
        data = next(iter(test_loader))
        image_in = data['image']
        labels = data['data']
        if use_cuda == True:
            image_in = image_in.cuda()
        logits = self.encoder_y.forward(image_in)
        target = labels.cpu().numpy()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        total_count = np.sum(target, axis=1)
        probs_target = target / total_count[:, None]
        rms =  np.sqrt((probs - probs_target)**2)
        return np.sum(rms) / labels.shape[0]

    
def train_ss(svi, train_s_loader, train_us_loader, use_cuda=False):
    # trains for one single epoch and returns normalised loss for one epoch
    # initialize loss accumulator
    epoch_loss_s = 0.
    epoch_loss_us = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    zip_list = zip(train_s_loader, cycle(train_us_loader)) if len(train_s_loader) > len(train_us_loader) else zip(cycle(train_s_loader), train_us_loader)
    for data_sup, data_unsup in zip_list:
        xs = data_sup['image']
        ys = data_sup['data']
        xus = data_unsup['image']
        yus = data_unsup['data']
        
#        ys = torch.zeros(ys.shape)
#        ys[:,0] = 1
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            xs = xs.cuda()
            xus = xus.cuda()
            # not really sure what Im doing here and not sure if necessary 
            ys = ys.cuda()
        # feeding in data. At times, omit labels
        # TODO seperate data set tolabelled and unlabelled rather than alternating as below
        batch_loss_s = svi.step(xs, ys)
        epoch_loss_s += batch_loss_s
        batch_loss_us = svi.step(xus)
        epoch_loss_us += batch_loss_us


    # return epoch loss
    normalizer_train_s = len(train_s_loader.dataset)
    total_epoch_loss_s = epoch_loss_s / normalizer_train_s
    normalizer_train_us = len(train_us_loader.dataset)
    total_epoch_loss_us = epoch_loss_us /normalizer_train_us
    return total_epoch_loss_s + total_epoch_loss_us

def evaluate(svi, test_s_loader, test_us_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for data_sup, data_unsup in zip(cycle(test_s_loader), test_us_loader):
        xs = data_sup['image']
        ys = data_sup['data']
        xus = data_unsup['image']
        yus = data_unsup['data']
        # if on GPU put mini-batch into CUDA memory
        batch_size = xs.size(0)

        if use_cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            xus = xus.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(xus)
        
        test_loss += svi.evaluate_loss(xs, ys)

        
    normalizer_test = len(test_s_loader.dataset) + len(test_us_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

def train_log(dir_name, ssvae, svi, train_s_loader, train_us_loader,
              test_s_loader, test_us_loader, img_len, num_epochs, plot_img_freq=1,
              num_img_plt=40, checkpoint_freq=10, use_cuda=True, test_freq=1, testing=False):
    writer = SummaryWriter("tb_data_all/" + dir_name)
    if not os.path.exists("checkpoints/" + dir_name):
        os.makedirs("checkpoints/" + dir_name)
        
    for epoch in range(num_epochs):
        total_epoch_loss_train = train_ss(svi, train_s_loader, train_us_loader, use_cuda=use_cuda)
        if epoch % test_freq == 0:
            print("Train loss", total_epoch_loss_train)
            writer.add_scalar('Train loss', total_epoch_loss_train, epoch)
            test_loss = evaluate(svi, test_s_loader, test_us_loader, use_cuda=use_cuda)
            writer.add_scalar('test loss', test_loss, epoch)
            print("test loss", test_loss)
            test_acc = ssvae.test_acc(test_us_loader, use_cuda=use_cuda)
            train_acc = ssvae.test_acc(train_us_loader, use_cuda=use_cuda)
            rms = ssvae.rms_calc(test_us_loader, use_cuda=use_cuda)
            print("rms:", rms, epoch)
            print("test accuracy:", test_acc, epoch)
            print("train accuracy:", train_acc, epoch)
            writer.add_scalar('test accuracy', test_acc, epoch)
            writer.add_scalar('train accuracy', train_acc, epoch)
            writer.add_scalar('rms', rms, epoch)
            
        if epoch % plot_img_freq == 0:
            data  = next(iter(test_us_loader))
            images_in = data['image'][0:num_img_plt]
            labels = data['data'][0:num_img_plt]
            img_grid_in = tv.utils.make_grid(images_in)
            images_out = ssvae.sample_img(images_in, labels, img_len, use_cuda=use_cuda)
            img_grid = tv.utils.make_grid(images_out)
            writer.add_image('images out from epoch ' + str(epoch), img_grid)
            writer.add_image('images in from epoch ' + str(epoch), img_grid_in)
            
        if epoch % checkpoint_freq == 0:
            torch.save(ssvae.encoder_y.state_dict(), "checkpoints/" + dir_name + "/encoder_y.checkpoint")
            torch.save(ssvae.encoder_z.state_dict(), "checkpoints/" + dir_name +  "/encoder_z.checkpoint")
            torch.save(ssvae.decoder.state_dict(), "checkpoints/" + dir_name +  "/decoder.checkpoint")
        writer.close()
