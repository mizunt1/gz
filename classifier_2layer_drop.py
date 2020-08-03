import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim=100, hidden=128, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.softmax(x)
        x = self.drop(x)
        return x
