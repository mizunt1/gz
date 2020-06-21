import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, z_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        return x
#        return nn.functional.log_softmax(hidden3, dim=1)


def test(classifier, test_loader, encoder=False, transform=False):
    correct = 0
    for x, y in test_loader:
        if transform != False:
            x = transform(x)
        if encoder != False:
            z_loc, z_scale = encoder(x)
            x = torch.cat((z_loc, z_scale), 1)
        output = classifier.forward(x)
        pred = output.argmax(dim=1)
        correct += pred.eq(y.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

def train_classifier(classifier, train_loader, test_loader, num_epochs=40, transform=False, encoder=False, use_cuda=False):
    # initialize loss accumulator
    running_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(num_epochs):
        i = 0
        for x, y in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            if transform != False:
                x = transform(x)
            optimizer.zero_grad()
            if encoder != False:
                z_loc, z_scale = encoder(x)
                combined_z = torch.cat((z_loc, z_scale), 1)
                x = combined_z

            outputs = classifier.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i+=1
            if i % 20 == 0:    # print every 2000 mini-batches
                print("loss is", loss.item())
                accuracy = test(classifier, test_loader, encoder=encoder, transform=transform)
                print("test accuracy", accuracy)
            
        i+=1


