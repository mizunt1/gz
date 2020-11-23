import torch
import random
import torchvision.datasets as dset
from torch.utils.data import IterableDataset
import torchvision as tv
from PIL import Image
import numpy as np

def flatten(x):
    x = np.array(x)
    return x.reshape(784)

def setup_data_loaders(data_type="digits", batch_size=128, use_cuda=False, root=None):
    if root == None:
        root = './data'
    download = True
    trans = tv.transforms.ToTensor()
    if data_type == "digits":
        train_set = dset.MNIST(root=root, train=True, transform=trans,
                               download=download)
        test_set = dset.MNIST(root=root, train=False, transform=trans)
    if data_type == "fashion":
        train_set = dset.FashionMNIST(root=root, train=True, transform=trans,
                               download=download)
        test_set = dset.FashionMNIST(root=root, train=False, transform=trans)


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
    train_indices = list(i for i in range(num_tests,len_data))
    test_set = torch.utils.data.Subset(data, test_indices)
    train_set = torch.utils.data.Subset(data, train_indices)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size)
    return train_loader, test_loader

def to_one_hot(x, batch_size):
    y = np.zeros(10)
    y[x] = 1.0
    return y.astype(np.float32)
    
def return_ss_loader(us_portion, batch_size, use_cuda=False, root=None, data_type="digits", transforms_list=None):
    if root == None:
        root = './data'
    download = True
    one_hot = tv.transforms.Lambda(lambda x: to_one_hot(x, batch_size))

    if transforms_list != None:
        trans = tv.transforms.Compose([tv.transforms.ToTensor(), *transforms_list])
    else:
        trans = tv.transforms.ToTensor()
    if data_type == "digits":
        train_set = dset.MNIST(root=root, train=True, transform=trans, target_transform=one_hot,
                               download=download)
        test_set = dset.MNIST(root=root, train=False, transform=trans, target_transform=one_hot)
    if data_type == "fashion":
        train_set = dset.FashionMNIST(root=root, train=True, transform=trans,
                               download=download)
        test_set = dset.FashionMNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    len_train = len(train_set)
    len_test = len(test_set)

    
    unsupervised_tests = round(len_test * us_portion)
    supervised_tests = len_test - unsupervised_tests
    
    unsupervised_train = round(len_train * us_portion)
    supervised_train = len_train - unsupervised_train
    
    # lists for test
    test_unsup = list(i for i in range(0, unsupervised_tests))
    test_supervised = list(i for i in range(unsupervised_tests,len_test))

    # lists for train
    train_unsup = list(i for i in range(0, unsupervised_train))
    train_supervised = list(i for i in range(unsupervised_train, len_train))


    test_s_set = torch.utils.data.Subset(test_set, test_supervised)
    test_us_set = torch.utils.data.Subset(test_set, test_unsup)
    train_s_set = torch.utils.data.Subset(train_set, train_supervised)
    train_us_set = torch.utils.data.Subset(train_set, train_unsup)

    test_s_loader = torch.utils.data.DataLoader(dataset=test_s_set, batch_size=batch_size, **kwargs)
    test_us_loader = torch.utils.data.DataLoader(dataset=test_us_set, batch_size=batch_size, **kwargs)
    train_s_loader = torch.utils.data.DataLoader(dataset=train_s_set, batch_size=batch_size, **kwargs)
    train_us_loader = torch.utils.data.DataLoader(dataset=train_us_set, batch_size=batch_size, **kwargs)

    return test_s_loader, test_us_loader, train_s_loader, train_us_loader


if __name__ == "__main__":
    a, b, c, d = return_ss_loader(0.8, 100)
