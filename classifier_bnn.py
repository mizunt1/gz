from pyro.nn import PyroSample, PyroModule
import torch
import pyro
import pyro.distributions as D
from pyro.infer import Predictive


def random_weight_helper(size, device):
    return PyroSample(D.Normal(torch.zeros(size, device=device), torch.ones(size, device=device)).to_event(len(size)))

class ClassifierBnn(PyroModule):
    def __init__(self, in_dim = 100, num_hidden = 128, num_out = 3, prior_std = 1., use_cuda=False):
        super(ClassifierBnn, self).__init__()
        
        # Define layers
        
        # linear layer 1
        self.linear_layer = PyroModule[torch.nn.Linear](in_dim, num_hidden)
        
        # linear alyer parameters as random variables
        self.linear_layer.weight = PyroSample(D.Normal(0., 1.).expand([num_hidden, in_dim]).to_event(1))
        device = torch.device('cuda') if use_cuda else None
        self.linear_layer.weight = random_weight_helper([num_hidden, in_dim], device)

        self.linear_layer.bias = random_weight_helper([num_hidden], device)


        # linear layer 2
        # output dimension is 3 because of the number of classes
        self.output_layer = PyroModule[torch.nn.Linear](num_hidden, num_out)
        
        # linear alyer parameters as random variables
        self.output_layer.weight = random_weight_helper([num_out, num_hidden], device)
        self.output_layer.bias = random_weight_helper([num_out], device)
        # activation function
        #self.activation = torch.nn.functional.softmax()

    def forward(self, x, y = None):            
        # latent variable
        
        z = self.linear_layer(x)
        z = torch.nn.functional.relu(z)
        z = self.output_layer(z)
        z = torch.nn.functional.softmax(z, dim=1)
        # likelihood
        with pyro.plate("data",size = x.shape[0], dim = -1):
            # I think this means each batch is independent            
            # z is the input to the distribution (categorical)
            obs = pyro.sample("obs",D.Multinomial(probs = z), obs=y)
        # return latent variable
        return z

#class ClassifierBnn(PyroModule):
    
#    def __init__(self, in_dim=100m num_hidden=200, num_out=3)

def predict(x, model, guide, num_samples=30):
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    # for a single image, output a mean and sd for each multivariate answer?
    
    yhats = predictive(x)["obs"].double()
    # yhats[0] seems to be integers 0 to 9, len 256
    # prediction for one model, for all items in batch
    # 20, 256
    mean = torch.mean(yhats, axis=0)
    std = torch.std(yhats.float(), 0).cpu().numpy()
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
        

