from load_mnist import setup_data_loaders, transform, return_data_loader
from torch.utils.tensorboard import SummaryWriter
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO
import os
import torchvision
import torch.nn as nn
import torch
import pyro
import argparse
parser = argparse.ArgumentParser()

class Encoder_y(nn.Module):
    # outputs a y given an x.
    # the classifier. distribution for y given an input x
    # input dim is whatever the input image size is,
    # output will be the probabilities a that parameterise y ~ cat(a)
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, output_size)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.softplus(y)
        y = self.softmax(y)
        return y


class Encoder_z(nn.Module):
    # input a x and a y, outputs a z
    # input x and y as flattened vector
    # inputsize should therefore be len(x) + len(y)
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc31 = nn.Linear(200, output_size)
        self.fc32 = nn.Linear(200, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # print("encoder z shape", x[0].shape, x[1].shape)# 
        y = x[1]
        x = x[0]
        if len(y.shape) > 2:
            """
            y is enumerated, so need to fix cat
            this is very not robust
            """
            x = x.expand(y.shape[0], *x.shape)
        x = torch.cat((x, y),-1)

        z = self.fc1(x)
        z = self.fc2(z)
        z = self.softplus(z)
        z_loc = self.fc31(z)
        z_scale = torch.exp(self.fc32(z))
        return z_loc, z_scale    



class Decoder(nn.Module):
    # takes y and z and outputs a x
    # input shape is therefore y and z concatenated
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 500)
        self.fc3 = nn.Linear(500, output_size)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        # print("decoder shape", z[0].shape, z[1].shape)# 
        y = z[1]
        z = z[0]
        x = torch.cat((z, y),-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.softplus(x)
        x = self.sigmoid(x)
        return x


class SSVAE(nn.Module):
    def __init__(self, input_size=784, output_size_y=10, output_size_z=100, output_size_d=784, use_cuda=False):
        super().__init__()
        self.output_size_y = output_size_y
        self.output_size_z = output_size_z
        self.input_size_z = output_size_y + input_size
        self.decoder_in_size = self.output_size_z + output_size_y
        self.encoder_y = Encoder_y(input_size, self.output_size_y)
        self.encoder_z = Encoder_z(self.input_size_z, self.output_size_z)
        self.decoder = Decoder(self.decoder_in_size, output_size_d)        
        if use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)
        batch_size = xs.size(0)

            # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # sample the handwriting style from the constant prior distribution
            prior_loc = xs.new_zeros([batch_size, self.output_size_z])
            prior_scale = xs.new_ones([batch_size, self.output_size_z])
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))
            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = xs.new_ones([batch_size, self.output_size_y]) / (1.0 * self.output_size_y)
            # vector of probabilities for each class, i.e. output_size
            # its a uniform prior
            # making labels one hot for onehotcat
            # not sure if this correct, maybe there is a better way as lewis
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
            # print("model, y sample", ys.shape)# 
            # print("model batch", dist.OneHotCategorical(alpha_prior).batch_shape)# 
            # one of the categories will be sampled, according to the distribution specified by alpha prior    
            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            loc = self.decoder.forward([zs, ys])
            # decoder networks takes a category, and a latent variable and outputs an observation x.
            pyro.sample("x", dist.Bernoulli(loc).to_event(1), obs=xs)

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
                # print("alpha", alpha.shape)# 
                # print("xs", xs.shape)# 
                # then we sample a classification using this parameterisation of the classifier.
                # the classifier is also like a generative model, where given the latents alpha, we 
                # output an observation y
                # and the latents alpha are given by an encoder
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))
                # print("guide ys sample", ys.shape)# 
                # print("ys guide batch shape",dist.OneHotCategorical(alpha).batch_shape)# 
                # print("y", ys[0][0])# 
                # if the labels y is known, then we dont have to sample from the above,
                # we just feed the actual y in to the encoder that takes x and y.
        
                # sample (and score) the latent handwriting-style with the variational
                # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            # change ys to one hot should do this somewhere else TODO

            loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def reconstruct_img(self, x, y, use_cuda=False):
        # encode image x
        batch_size = x.size(0)
        y = y.reshape(batch_size, 1)
        y = (y == torch.arange(10).reshape(1, 10)).float()
        x = transform(x)
        if use_cuda == True:
            x = x.cuda()
            y = y.cuda()
        z_loc, z_scale = self.encoder_z([x,y])
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder([z,y])
        return loc_img.reshape([batch_size, 1, 28, 28])

def train_ss(svi, train_loader, use_cuda=False, transform=False):
    # trains for one single epoch and returns normalised loss for one epoch
    labelled = True
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, y in train_loader:
        batch_size = x.size(0)
        # changing labels to one hot encoding
        # I think this is necessary when using dist.OneHotCategorical but not sure 
        y = y.reshape(batch_size, 1)
        y = (y == torch.arange(10).reshape(1, 10)).float()
        if transform != False:
            # flattens images to 1d vector
            x = transform(x)
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            # not really sure what Im doing here and not sure if necessary 
            y = y.cuda()
        # feeding in data. At times, omit labels
        # TODO seperate data set tolabelled and unlabelled rather than alternating as below
        if labelled == True:
            batch_loss = svi.step(x, y)
            epoch_loss += batch_loss
            labelled = False
        else:
            batch_loss = svi.step(x)
            epoch_loss += batch_loss
            labelled = True
    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False, transform=transform):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x, y in test_loader:
        # if on GPU put mini-batch into CUDA memory
        batch_size = x.size(0)
        y = y.reshape(batch_size, 1)
        y = (y == torch.arange(10).reshape(1, 10)).float()
        
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        if transform != False:
            x = transform(x)
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)

    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test

parser.add_argument('--data_save', default="tb_data/")
parser.add_argument('--data_load', default="/scratch-ssd/oatml/data")
parser.add_argument('--data_type', default="digits")
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--no_cuda', default=False, action='store_true')
parser.add_argument('--checkpoint_dir', default="checkpoints/")
args = parser.parse_args()
writer = SummaryWriter(args.data_save)
pyro.clear_param_store()
print("loading data")
print("data load", args.data_load)

use_cuda = not args.no_cuda
train_loader, test_loader = setup_data_loaders(data_type=args.data_type, batch_size=72, root=args.data_load, use_cuda=use_cuda)
ssvae = SSVAE(use_cuda=use_cuda)
optimizer = Adam({"lr": 3.0e-4})
#svi = SVI(ssvae.model, ssvae.guide, optimizer, loss=Trace_ELBO())
guide = config_enumerate(ssvae.guide, "parallel", expand=True)
svi = SVI(ssvae.model, guide, optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

print("start train")
num_epochs = args.epochs
for epoch in range(num_epochs):
    total_epoch_loss_train = train_ss(svi, train_loader, use_cuda=use_cuda, transform=transform)
    print("epoch loss", total_epoch_loss_train)
    writer.add_scalar('Train loss', total_epoch_loss_train, epoch)
    if epoch % 2 == 0:
        test_loss = evaluate(svi, test_loader, use_cuda=use_cuda, transform=transform)
        writer.add_scalar('test loss', test_loss, epoch)
        print("test loss", test_loss)
        

train_loader, test_loader = setup_data_loaders(data_type=args.data_type, batch_size=9, root=args.data_load, use_cuda=use_cuda)
images, labels = next(iter(train_loader))
images_out = ssvae.reconstruct_img(images, labels, use_cuda=use_cuda)
img_grid = torchvision.utils.make_grid(images_out)
writer.add_image('images', img_grid)
writer.close()
os.makedirs(args.checkpoint_dir)
torch.save(ssvae.encoder_y.state_dict(), args.checkpoint_dir + "/encoder_y.checkpoint")
torch.save(ssvae.encoder_z.state_dict(),  args.checkpoint_dir +  "/encoder_z.checkpoint")
torch.save(ssvae.decoder.state_dict(),  args.checkpoint_dir +  "/decoder.checkpoint")
