import torch.nn as nn
from layers import LinearBlock

class Classifier(nn.Module):
    def __init__(self, in_dim=100, hidden=200, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            LinearBlock(hidden),
            nn.Linear(hidden, int(hidden/2)),
            LinearBlock(int(hidden/2)),
            nn.Linear(int(hidden/2), out_dim),
            nn.ELU()
        )


    def forward(self, x):
        return self.net(x)

