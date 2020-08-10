
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import torch
import torch.distributions as D
from layers import Reshape

class Classifier(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # first conv pair
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            Reshape(-1,int(in_dim/2)),
            nn.Dropout(),
            nn.Linear(int(in_dim/2), hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1),
            # second conv pair
        )

    def forward(self, x):
        return self.net(x)

    
if __name__ == "__main__":
    data = torch.zeros([3, 16, 8, 8])
    model = Classifier()
    out = model(data)
    print(out.shape)
