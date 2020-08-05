from pyro.nn import PyroSample, PyroModule
from pyro.distributions import Normal, Categorical
import torch
import pyro
import pyro.distributions as D
from pyro.infer import Predictive
class ClassifierBnn(PyroModule):
    
    def __init__(self, in_dim = 100, num_hidden = 200, num_out = 3, prior_std = 1.):
        
        # call to father constructor
        super().__init__()
        
        # define prior
        prior = Normal(0, prior_std)
        
        # Define layers
        
        # linear layer 1
        self.linear_layer = PyroModule[torch.nn.Linear](in_dim, num_hidden)
        
        # linear alyer parameters as random variables
        self.linear_layer.weights = PyroSample(prior.expand([num_hidden, in_dim]).to_event(2))
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
        z = torch.nn.functional.softmax(z, dim=1)
        # likelihood
        with pyro.plate("data",size = x.shape[0], dim = -1):
            # I think this means each batch is independent            
            # z is the input to the distribution (categorical)
            obs = pyro.sample("obs",D.Multinomial(probs = z), obs=y)
        # return latent variable
        return z


def predict(x, model, guide, num_samples=30):
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    # for a single image, output a mean and sd for each multivariate answer?
    
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
        

