import math

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor):
        up_blk_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.blk1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU())
        blk26 = [ResBlk(64) for k in range(5)]
        self.blk26 = nn.Sequential(*blk26)
        self.blk7 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.PReLU())
        blk8 = [UpBlk(64, 2) for _ in range(up_blk_num)]
        blk8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.blk8 = nn.Sequential(*blk8)


    def forward(self, x):
        blk1 = self.blk1(x)
    # blk26 = list()
    # blk26.append(self.blk26[0](blk1))
    # for k in range(1, 5):
    #     blk26.append(self.blk26[k](blk26[k-1]))
        blk26 = self.blk26(blk1)
     #   blk3 = self.blk26[1](blk2)
     #   blk4 = self.blk26[2](blk3)
     #   blk5 = self.blk26[3](blk4)
     #   blk6 = self.blk26[4](blk5)
        blk7 = self.blk7(blk26)
        blk8 = self.blk8(blk1 + blk7)

        return (torch.tanh(blk8) + 1) / 2

class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()

        type321 = lambda kn: [nn.Conv2d(kn, kn, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(kn), nn.LeakyReLU(0.2)]
        type31 = lambda kn: [nn.Conv2d(kn, kn*2, kernel_size=3, padding=1), nn.BatchNorm2d(kn*2), nn.LeakyReLU(0.2)]

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            *type321(64),
            *type31(64),
            *type321(128), 
            *type31(128),
            *type321(256),
            *type31(256),
            *type321(512),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        b_size = x.size(0)
        return torch.sigmoid(self.net(x).view(b_size))


class ResBlk(nn.Module):
    def __init__(self, channels):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)

        return x + res


class UpBlk(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
