import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim=100, hidden=200, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.softmax(x)
        return x
