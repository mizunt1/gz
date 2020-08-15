
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
import torch
import torch.distributions as D
from layers import Reshape

class Mike(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # first conv pair
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # second conv pair
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # third conv
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # fourth conv
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # linear
            Reshape(-1, 1024),
            nn.Linear(1024, 128),
            nn.ReLU(),
            # linear
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x - 0.222
        x = x / 0.156
        return self.net(x)

def multinomial_loss(probs, values):
    return torch.sum(-1 * D.Multinomial(probs=probs).log_prob(values.float()))
            
def train(train_loader, classifier,optim, use_cuda=False):
    """
    trains for one epoch
    """
    loss = multinomial_loss
    running_loss = 0
    optimizer= optim
    num_steps = 0
    for data in train_loader:
        x = data['image']
        y = data['data']
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        x_out = classifier(x)
        classifier_loss = loss(x_out, y)
        classifier_loss.backward()
        optimizer.step()
        num_steps += 1
        running_loss += classifier_loss.item()
    return running_loss / len(train_loader.dataset), num_steps


def rms_calc(probs, target):
    """
    total rms for a single batch
    """
    probs = probs.detach().cpu().numpy()
    target = target.cpu().numpy()
    total_count = np.sum(target, axis=1)
    probs_target = target / total_count[:, None]
    rms =  np.sqrt((probs - probs_target)**2)
    return np.sum(rms)

def evaluate(test_loader, classifier, use_cuda=False):
    loss = multinomial_loss        
    running_loss = 0
    running_rms = 0
    for data in test_loader:
        x = data['image']
        y = data['data']
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        x_out = classifier(x)
        running_loss += loss(x_out,y).item()
        running_rms += rms_calc(x_out, y)
        av_loss = running_loss / len(test_loader.dataset)
        rms = running_rms / len(test_loader.dataset)
    return av_loss, rms

def train_log(dir_name, classifier, optim, train_loader, test_loader, test_freq=1,
              num_epochs=10, use_cuda=False):
    num_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total_steps = 0
    print("num params: ", num_params)
    writer = SummaryWriter("tb_data_all/" + dir_name)
    if use_cuda:
        classifier.cuda()
    for epoch in range(num_epochs):
        print("training")
        train_loss, num_steps = train(train_loader, classifier,optim, use_cuda=use_cuda)
        total_steps += num_steps
        print("end train")
        print("[epoch %03d]  average training loss: %.4f" % (total_steps, train_loss))
        writer.add_scalar("Train loss classifier", train_loss, total_steps)
        if epoch % test_freq == 0:
            print("evaluating")
            eval_loss, rms = evaluate(test_loader, classifier, use_cuda=use_cuda)
            print("[epoch %03d] average test_loss: %.4f" % (total_steps, eval_loss))
            print("[epoch %03d] average rms: %.4f" % (total_steps, rms))
            writer.add_scalar("Test loss classifier", eval_loss, total_steps)
            writer.add_scalar("rms normalised", rms, total_steps)
            
    writer.close()
    
if __name__ == "__main__":
    data = torch.zeros([12, 1, 128, 128])
    model = Mike()
    out = model(data)
    print(out.shape)
