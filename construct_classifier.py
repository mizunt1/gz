import torch.optim as optim
class Classifier(nn.Module):
    def __init__(self, z_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(z_dim*2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
    
    def forward(self, x):
        hidden1 = self.fc1(x)
        hidden2 = self.fc2(hidden1)
        hidden3 = self.fc3(hidden2)
        return nn.functional.log_softmax(hidden3, dim=1)

def test(encoder, classifier, test_loader):
    correct = 0
    for data, target in test_loader:
        z_loc, z_scale = vae.encoder(data)
        combined_z = torch.cat((z_loc, z_scale), 1)
    
        output = classifier.forward(combined_z)
        pred = output.argmax(dim=1)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    return correct / len(test_loader.dataset)

def train_classifier(train_loader, test_loader, encoder=False, use_cuda=False):
    # initialize loss accumulator
    running_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    classifier=Classifier(50,200)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        i = 0
        for x, y in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if use_cuda:
                x = x.cuda()
            optimizer.zero_grad()
            z_loc, z_scale = encoder(x)
            combined_z = torch.cat((z_loc, z_scale), 1)
            outputs = classifier.forward(combined_z)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if epoch % 20 == 0:    # print every 2000 mini-batches
                print(loss.item())
                accuracy = test(vae.encoder, classifier, test_loader)
                print("test accuracy", accuracy)
