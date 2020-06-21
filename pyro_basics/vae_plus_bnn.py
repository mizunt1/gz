#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
import pyro.contrib.examples.util  # patches torchvision
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)
# Enable smoke test - run the notebook cells on CI.
#smoke_test = 'CI' in os.environ

def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


# In[4]:


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img


# In[5]:


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


# In[6]:


# define the model p(x|z)p(z)
def model(self, x):
    # register PyTorch module `decoder` with Pyro
    pyro.module("decoder", self.decoder)
    with pyro.plate("data", x.shape[0]):
        # setup hyperparameters for prior p(z)
        z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
        # z loc torch.Size([256, 50])
        z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
        # we sample a Z from a (0, I) normal distribution
        # then we pass it though a nn
        # mu = nn(z)
        # then this mu is used in another dist
        # p(x|z) where z is samples
        # then we sample an x from this
        # the idea is, this nn function learns a distribution
        # that is, what would it be like to sample z from P(z|X)
        # 
        # z shape torch.Size([256, 50])
        # decode the latent code z
        loc_img = self.decoder.forward(z)
        #loc img torch.Size([256, 784])
        # score against actual images
        # bern shape Independent(Bernoulli(probs: torch.Size([256, 784])), 1)
        # 784 is the batch size
        # 256 is the image size
        pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))


# In[7]:


# define the guide (i.e. variational distribution) q(z|x)
def guide(self, x):
    # register PyTorch module `encoder` with Pyro
    pyro.module("encoder", self.encoder)
    with pyro.plate("data", x.shape[0]):
        # use the encoder to get the parameters used to define q(z|x)
        z_loc, z_scale = self.encoder.forward(x)
        # p(z,b) = q(b)mult(i=1 to i=N)q(zi|f(xi))
        
        # given an image, we output a distribution for z
        # then we sample a z. because the guide always gives the
        # approximate posterior, the variational inference
        # sample the latent code z
        pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


# In[8]:


class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    
    
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            # decoder is where the image goes 
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img


# In[9]:


vae = VAE()

optimizer = Adam({"lr": 1.0e-3})

svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, _ in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=False):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x, _ in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


# In[10]:


LEARNING_RATE = 1.0e-3
USE_CUDA = False
smoke_test = False

# Run only for a single iteration for testing
NUM_EPOCHS = 250
TEST_FREQUENCY = 5
train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=USE_CUDA)

# clear param store
pyro.clear_param_store()

# setup the VAE
vae = VAE(use_cuda=USE_CUDA)

# setup the optimizer
adam_args = {"lr": LEARNING_RATE}
optimizer = Adam(adam_args)

# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

train_elbo = []
test_elbo = []
# training loop
print("started train")
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_loader, use_cuda=USE_CUDA)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=USE_CUDA)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))
print("end train")

torch.save(vae.encoder.state_dict(), "encoder.checkpoint")
torch.save(vae.decoder.state_dict(), "decoder.checkpoint")

print("saved")

from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, Categorical

class ClassifierBnn(PyroModule):
    
    def __init__(self, num_in = 100, num_hidden = 200, num_out = 10, prior_std = 1.):
        
        # call to father constructor
        super().__init__()
        
        # define prior
        prior = Normal(0, prior_std)
        
        # Define layers
        
        # linear layer 1
        self.linear_layer = PyroModule[torch.nn.Linear](num_in, num_hidden)
        
        # linear alyer parameters as random variables
        self.linear_layer.weights = PyroSample(prior.expand([num_hidden, num_in]).to_event(2))
        self.linear_layer.bias = PyroSample(prior.expand([num_hidden]).to_event(1))
        
        # linear layer 2
        # output dimension is 3 because of the number of classes
        self.output_layer = PyroModule[torch.nn.Linear](num_hidden, num_out)
        
        # linear alyer parameters as random variables
        self.output_layer.weights = PyroSample(prior.expand([num_out, num_hidden]).to_event(2))
        self.output_layer.bias = PyroSample(prior.expand([num_out]).to_event(1))
        
        # activation function
        #self.activation = torch.nn.functional.softmax()
        
    def forward(self, x, y = None):
            
        # latent variable
        z = self.linear_layer(x)
        z = self.output_layer(z)
        z = torch.nn.functional.log_softmax(z, dim=1)
        # likelihood
        with pyro.plate("data",size = x.shape[0], dim = -1):
            # I think this means each batch is independent            
            # z is the input to the distribution (categorical)
            obs = pyro.sample("obs", Categorical(logits = z), obs=y)
        # return latent variable
        return z



# validate NN

pyro.enable_validation(True)

model = ClassifierBnn()
x, y = next(iter(train_loader))
z_loc, z_scale = vae.encoder(x)
combined_z = torch.cat((z_loc, z_scale), 1)


print(pyro.poutine.trace(model).get_trace(combined_z, y).format_shapes())


# In[13]:


pyro.enable_validation(True)
pyro.clear_param_store()
model = ClassifierBnn(num_hidden = 10, prior_std = 1.)

# define guide
from pyro.infer.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model, init_scale=1e-1)

# define SVI (model for training)
svi = pyro.infer.SVI(model,
                    guide,
                    optim=pyro.optim.ClippedAdam({'lr':1e-3}),
                    # Define conventional ELBO
                     loss=pyro.infer.Trace_ELBO())


# In[14]:


from pyro.infer import Predictive
predictive = Predictive(model, guide=guide, num_samples=20)

def predict(x):
    # for a single image, output a mean and sd for category
    yhats = predictive(x)["obs"].double()
    # yhats[0] seems to be integers 0 to 9, len 256
    # prediction for one model, for all items in batch
    # 20, 256
    mean = torch.mean(yhats, axis=0)
    std = torch.std(yhats.float(), 0).numpy()
    # yhats outputs a batch size number of predictions for 20 models
    # yhats seem to be a dictionary of weights
    return mean, std

def evaluate_test(test_loader, encoder):
    accuracy = 0
    for x, y in test_loader:
        z_loc, z_scale = encoder(x)
        combined_z = torch.cat((z_loc, z_scale), 1)
        mean, std = predict(combined_z)
        num_correct_in_batch = torch.sum(torch.eq(mean.int(),y))
        accuracy += num_correct_in_batch.numpy()/len(y)
    return accuracy / (len(test_loader))
        

num_epochs = 1000

# Define number of epochs
epoch_loss = np.zeros(shape=(num_epochs,))

test_freq = 10
# training
bnn_train_epoch = 150
for epoch in range(150):
    i = 0
    for x, y in train_loader:
        i +=1
        # batches of size 256 are being fed in 
        z_loc, z_scale = vae.encoder(x)
        combined_z = torch.cat((z_loc, z_scale), 1)
        loss = svi.step(combined_z, y)
        if i % test_freq == 0:
            test_acc = evaluate_test(test_loader, vae.encoder)
            print("test acc", test_acc)
            print("loss", loss)
            print("mean", mean[0], "y is", y[0])
            print("train acc", accuracy_per_batch)
        mean, std = predict(combined_z)
        accuracy_per_batch = torch.sum(torch.eq(mean.int(),y)).numpy()/len(y)





