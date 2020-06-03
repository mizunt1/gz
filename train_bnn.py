from bnn import ClassifierBnn, train_bnn, validate_model
from pyro.infer import Predictive
from load_mnist import setup_data_loaders
from pyro.infer.autoguide import AutoDiagonalNormal
import torchvision.transforms as transforms

def transform(x):
    x = x.squeeze()
    batch = x.shape[0]
    x = x.reshape(batch,784)
    return x

train_loader, test_loader = setup_data_loaders()

model = ClassifierBnn(num_in=784)
guide = AutoDiagonalNormal(model, init_scale=1e-1)
train_bnn(20, train_loader, test_loader, model, guide, transform=transform)
