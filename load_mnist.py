import torch
import random
import torchvision.datasets as dset
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
from PIL import Image

def transform(x):
    x = x.squeeze()
    batch = x.shape[0]
    x = x.reshape(batch,784)
    return x

def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

def get_subset(data, indices, batch_size):
    data_new = torch.utils.data.Subset(data, indices)
    data_loader = torch.utils.data.DataLoader(data=data_new, batch_size=batch_size)
    return data_loader

def return_data_loader(data, test_proportion, batch_size):
    len_data = len(data)
    num_tests = int(len_data * test_proportion)
    test_indices = list(i for i in range(0,num_tests))
    train_indices = list(i for i in range(0, num_tests))
    test_set = torch.utils.data.Subset(data, test_indices)
    train_set = torch.utils.data.Subset(data, train_indices)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size)
    return train_loader, test_loader




if __name__ == "__main__":
    a, b = get_semi_supervised_data(100, 0.5, False)
