import torch
import random
import torchvision.datasets as dset
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


def get_semi_supervised_data(batch_size, proportion, use_cuda):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=download)
    nulls_train = int(len(train_set) * proportion)
    nulls_test = int(len(test_set) * proportion)
    train_set.targets[0:nulls_train] = -1
    test_set.targets[0:nulls_test] = -1
    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


class SemiSupervisedData(torch.utils.data.Dataset):
    def __init__(self):
        root = './data'
        download = True
        trans = transforms.ToTensor()
        self.mnist = dset.MNIST(root=root, transform=trans,
                               download=download)

    def __getitem__(self, index):
        img, target = self.mnist.data[index], int(self.mnist.targets[index])
        if bool(random.getrandbits(1)) == 1:
            return img, target
        else:
            return img, None
        

if __name__ == "__main__":
    a, b = get_semi_supervised_data(100, 0.5, False)
