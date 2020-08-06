import torch.nn as nn
import pyro
import pyro.distributions as D

class Classifier(nn.Module):
    def __init__(self, in_dim=100, hidden=128, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, y):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        with pyro.plate("data",size = x.shape[0], dim = -1):
            # I think this means each batch is independent            
            # z is the input to the distribution (categorical)
            obs = pyro.sample("obs",D.Multinomial(probs = x), obs=y)
        # return latent variable
        return x
