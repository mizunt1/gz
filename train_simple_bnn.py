import torch
import pyro
from bnn import ClassifierBnn, train_bnn, validate_model
from pyro.infer import Predictive
from load_mnist import setup_data_loaders

train_loader, test_loader = setup_data_loaders()

def transform(x):
    x = x.squeeze()
    batch = x.shape[0]
    x = x.reshape(batch,784)
    return x

model = ClassifierBnn(num_in=784)
pyro.enable_validation(True)
from pyro.infer.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model, init_scale=1e-1)
pyro.clear_param_store()
validate_model(train_loader, transform=transform)
train_bnn(100, train_loader, test_loader, model, guide, transform=transform)



