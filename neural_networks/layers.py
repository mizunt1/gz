from torch import nn
import torch
class Conv2dEnum(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias, stride=1):
        super(Conv2dEnum, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        length = len(x.shape)
        enum_shape = x.shape[:-3]
        x = torch.flatten(x, start_dim=0, end_dim=length -4)
        x = self.conv(x)
        normal_shape = x.shape[1:]
        x = x.reshape(*enum_shape, *normal_shape)
        return x

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
        length = len(x.shape)
        enum_shape = x.shape[:-3]
        x = torch.flatten(x, start_dim=0, end_dim=length -4)
        x = x + self.body(x)
        normal_shape = x.shape[1:]
        x = x.reshape(*enum_shape, *normal_shape)
        return x

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
        self.merge = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        length = len(x.shape)
        enum_shape = x.shape[:-3]
        x = torch.flatten(x, start_dim=0, end_dim=length -4)
        x = self.merge(self.skip(x) + self.body(x))
        normal_shape = x.shape[1:]
        x = x.reshape(*enum_shape, *normal_shape)
        return x

class BatchNorm2dEnum(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.batch = torch.nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        length = len(x.shape)
        enum_shape = x.shape[:-3]
        x = torch.flatten(x, start_dim=0, end_dim=length -4)
        x = self.batch(x)
        normal_shape = x.shape[1:]
        x = x.reshape(*enum_shape, *normal_shape)
        return x


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


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

    
if __name__ == "__main__":
    o = torch.zeros([2,3,10, 1, 80, 80])
    length = len(o.shape)
    enum_shape = o.shape[:-3]
    normal_shape = o.shape[3:]
    print("norm", normal_shape)
    print("enu", enum_shape)
    x = torch.flatten(o, start_dim=0, end_dim=length-4)
    print(x.shape)
    x = x.reshape(*enum_shape, *normal_shape)
    print(torch.all(torch.eq(o, x)))

    o = torch.zeros([2,3,10, 1, 80, 80])
    x = Conv2dEnum(1, 32, 3, 1, bias=False)(o)
    print(x.shape)
