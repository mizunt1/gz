from vae import VAE, setup_data_loaders
from simple_classifier import Classifier, train_classifier
import torch

train_loader, test_loader = setup_data_loaders()
vae = VAE()
encoder = vae.encoder
encoder.load_state_dict(torch.load("encoder.checkpoint"))
train_classifier(train_loader, test_loader, num_epochs=10, encoder=encoder)
