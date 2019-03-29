import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, num_features, scaling_factor=0.1):
        super(ResBlock, self).__init__()
        self.scaling_factor = scaling_factor
        self.feas = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )

    def forward(self, x):
        return x + self.scaling_factor*self.feas(x)


class PixShuffleBlock(nn.Module):
    def __init__(self, in_features, upscale_factor=2):
        super(PixShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_features, in_features*upscale_factor*upscale_factor, 3, padding=1)
        self.up = nn.PixelShuffle(upscale_factor)
        
    def forward(self, x):
        fea = self.conv(x)
        fea = self.up(fea)
        return fea


class EDSR(nn.Module):
    def __init__(self, num_resblocks, num_features, scaling_factor):
        super(EDSR, self).__init__()
        self.conv = nn.Conv2d(3, num_features, 3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_features) for i in range(num_resblocks)]
        )
        self.up_sampler = PixShuffleBlock(num_features, upscale_factor=scaling_factor)
        self.output = nn.Conv2d(num_features, 3, 3, padding=1)

    def forward(self, x):
        fea = self.conv(x)
        fea = self.res_blocks(fea)
        fea = self.up_sampler(fea)
        fea = self.output(fea)
        return fea


if __name__=="__main__":
    net = EDSR(32, 256, 2)
    net.cuda()
    lr = torch.rand(4,3,64,64).cuda()
    print(lr.shape)
    hr = net(lr)
    print(hr.shape)