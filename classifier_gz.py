import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim=100, hidden=50, out_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
    
    def forward(self, x):
        hidden1 = self.fc1(x)
        hidden2 = self.fc2(hidden1)
        hidden3 = self.fc3(hidden2)
        return nn.functional.log_softmax(hidden3, dim=1)
