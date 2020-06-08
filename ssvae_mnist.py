from load_mnist import setup_data_loaders, get_semi_supervised_data, transform, SemiSupervisedData
import pyro.distributions as dist
from vae import evaluate
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
import torch.nn as nn
import torch
import pyro

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
        self.softmax = nn.Softmax()

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.softplus(y)
        y = self.softmax(y, dim=1)
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
        x = torch.cat((x[0], x[1]), 1)
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
        x = torch.cat((z[0], z[1]), 1)
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
        self.input_size_z = self.output_size_y + self.output_size_z
        self.encoder_y = Encoder_y(input_size, self.output_size_y)
        self.encoder_z = Encoder_z(self.input_size_z, self.output_size_z)

        self.decoder = Decoder(self.input_size_z, output_size_d)


        
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
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
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
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))
                # if the labels y is known, then we dont have to sample from the above,
                # we just feed the actual y in to the encoder that takes x and y.
        
                # sample (and score) the latent handwriting-style with the variational
                # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

def validate_model(train_loader, model, encoder=False, transform=False):
    x, y = next(iter(train_loader))
    if transform != False:
        x = transform(x)

    #print(pyro.poutine.trace(model).get_trace(x, y).format_shapes())
    #pyro.enable_validation(True)
    #pyro.clear_param_store()

def train(svi, train_loader, use_cuda=False, transform=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for data in train_loader:
        if len(data) == 2:
            x = data[0]
            y = data[1]
        else:
            x = data
            y = None
        if transform != False:
            x = transform(x)
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x,y)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

if __name__ == "__main__":
    
    #train_loader, test_loader = setup_data_loaders()
    semi_sup_data = SemiSupervisedData()
    train_indices = list(i for i in range(0,800))
    test_indices = list(i for i in range(800, 1000))
    train_set = torch.utils.data.Subset(semi_sup_data, train_indices)
    test_set =  torch.utils.data.Subset(semi_sup_data, test_indices)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=20)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=20)
    ssvae = SSVAE()
    optimizer = Adam({"lr": 1.0e-3})

    #pyro.enable_validation(True)
    svi = SVI(ssvae.model, config_enumerate(ssvae.guide), optimizer, loss=TraceEnum_ELBO())
    for epoch in range(20):
        total_epoch_loss_train = train(svi, train_loader, use_cuda=False, transform=transform)
        print("epoch loss", total_epoch_loss_train)

        if epoch % 2 == 0:
            test_elbo = evaluate(svi, test_loader, use_cuda=False)