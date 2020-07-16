import torch


def ilogit(p):
    return torch.log(p + 1e-6) - torch.log(1 - p + 1e-6)
