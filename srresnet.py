import torch
from torch import nn
import math


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super(ResBlock, self).__init__()
        self.feas = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features), nn.PReLU(),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.BatchNorm2d(num_features))

    def forward(self, x):
        return x + self.feas(x)


class PixShuffleBlock(nn.Module):
    def __init__(self, in_features, upscale_factor=2):
        super(PixShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_features,
            in_features * upscale_factor * upscale_factor,
            3,
            padding=1)
        self.up = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        fea = self.conv(x)
        fea = self.up(fea)
        fea = self.prelu(fea)
        return fea


class SRResNet(nn.Module):
    def __init__(self, num_resblocks, num_features, scaling_factor):
        super(SRResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, num_features, 3, padding=1)
        self.prelu = nn.PReLU()
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_features) for i in range(num_resblocks)])
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_features)
        self.pix_blocks = nn.Sequential(*[
            PixShuffleBlock(num_features)
            for i in range(int(math.log2(scaling_factor)))
        ])
        self.conv3 = nn.Conv2d(num_features, 3, 1, padding=1)

    def forward(self, x):
        fea_0 = self.prelu(self.conv1(x))
        fea = self.res_blocks(fea_0)
        fea = fea_0 + self.bn(self.conv2(fea))
        fea = self.pix_blocks(fea)
        fea = self.conv3(fea)
        return fea


if __name__ == "__main__":
    net = SRResNet(6, 64, 4)
    x = torch.rand([6, 3, 64, 64])
    y = net(x)
    print(x.shape)
    print(y.shape)