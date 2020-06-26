from torch import nn
import torch
class ConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5, padding=2, bias=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.ELU()

        )
    def forward(self, x):
        return x + self.body(x)

class LinearBlock(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_size, in_size),
            nn.ELU(),
            nn.Linear(in_size, in_size),
            nn.ELU(),
            nn.Linear(in_size, in_size),
            nn.ELU()
        )
    def forward(self, x):
        return x + self.body(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UpResBloc(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=bias),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.skip = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.merge(self.skip(x) + self.body(x))


class DownResBloc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(in_channels),
            nn.MaxPool2d(2),
        )
        self.skip = nn.MaxPool2d(2, mode="nearest")
        self.merge = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.merge(self.skip(x) + self.body(x))

if __name__ == "__main__":
    x = torch.zeros([10, 1, 80, 80])
    
