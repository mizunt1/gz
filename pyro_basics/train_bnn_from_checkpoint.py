import torch
import pyro
from bnn import ClassifierBnn, train_bnn, validate_model
from pyro.infer import Predictive
from vae import setup_data_loaders, VAE

train_loader, test_loader = setup_data_loaders()
vae = VAE()
encoder = vae.encoder
encoder.load_state_dict(torch.load("encoder.checkpoint"))

model = ClassifierBnn()
pyro.enable_validation(True)
from pyro.infer.autoguide import AutoDiagonalNormal
guide = AutoDiagonalNormal(model, init_scale=1e-1)
pyro.clear_param_store()
validate_model(train_loader, encoder)
train_bnn(100, train_loader, test_loader, model, guide, encoder)



