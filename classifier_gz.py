import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim=100, hidden=200, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, int(hidden /2))
        self.fc4 = nn.Linear(int(hidden/2), out_dim)
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.elu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
