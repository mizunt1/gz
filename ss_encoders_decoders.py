import torch.nn as nn
import utils
import torch
class Encoder_y(nn.Module):
    # outputs a y given an x.
    # the classifier. distribution for y given an input x
    # input dim is whatever the input image size is,
    # output will be the probabilities a that parameterise y ~ cat(a)
    def __init__(self, input_size=784, output_size=10):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc2 = nn.Linear(400, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.elu = nn.ELU()

    def forward(self, x):
        x = x.view(-1, self.input_size)
        y = self.fc1(x)
        y = self.elu(y)
        y = self.fc2(y)
        y = self.softmax(y)
        return y


class Encoder_z(nn.Module):
    # input a x and a y, outputs a z
    # input x and y as flattened vector
    # inputsize should therefore be len(x) + len(y)
    def __init__(self, input_size=794, output_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc31 = nn.Linear(200, output_size)
        self.fc32 = nn.Linear(200, output_size)
        self.elu = nn.ELU()

    def forward(self, x):
        x = utils.cat(x[0], x[1] ,-1)
        z = self.fc1(x)
        z = self.elu(z)
        z = self.fc2(z)
        z_loc = self.fc31(z)
        z_scale = torch.exp(self.fc32(z))
        return z_loc, z_scale    

class Decoder(nn.Module):
    # takes y and z and outputs a x
    # input shape is therefore y and z concatenated
    def __init__(self, input_size=110, output_size=784):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 500)
        self.fc3 = nn.Linear(500, output_size)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()


    def forward(self, z):
        x = utils.cat(z[0], z[1], -1)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


