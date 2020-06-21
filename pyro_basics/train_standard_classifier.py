from load_mnist import setup_data_loaders
from simple_classifier import Classifier, train_classifier
def transform(x):
    x = x.squeeze()
    batch = x.shape[0]
    x = x.reshape(batch,784)
    return x

train_loader, test_loader = setup_data_loaders()
model = Classifier(784, 200)
train_classifier(model, train_loader, test_loader, transform=transform)
