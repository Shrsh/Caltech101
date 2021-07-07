import torch
import torch.nn as nn
import torch.nn.functional as F


class arch(nn.Module):

    def __init__(self, negative_slope: float):
        self.negative_slope = negative_slope
        super(arch, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(16, 64, 3, 1, 1)

        self.up_image = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.up = torch.nn.ConvTranspose2d(64, 64, stride=4, kernel_size=4)

        self.block1 = ResNetBlock(kernel_size=3, negative_slope=self.negative_slope)
        self.block2 = ResNetBlock(kernel_size=3, negative_slope=self.negative_slope)
        self.block3 = ResNetBlock(kernel_size=3, negative_slope=self.negative_slope)
        self.block4 = ResNetBlock(kernel_size=3, negative_slope=self.negative_slope)

        self.conv3 = torch.nn.Conv2d(64, 16, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(16, 3, 1, 1, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        LR_feat = F.leaky_relu(self.conv1(input), negative_slope=self.negative_slope)
        LR_feat = F.leaky_relu(self.conv2(LR_feat), negative_slope=self.negative_slope)

        out1 = self.block1(LR_feat)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        out5 = F.leaky_relu(self.conv3(self.up(out4)), negative_slope=0.2)
        out6 = self.conv4(out5)

        return torch.add(out6, self.up_image(input))


class ResNetBlock(nn.Module):
    r"""Resnet block structure"""

    def __init__(self, in_channels: int = 64, out_channels: int = 64, kernel_size=3,
                 negative_slope=0.2, dilation=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,
                      padding=int((kernel_size * dilation - 1) / 2), dilation=dilation),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                      padding=int((kernel_size * dilation - 1) / 2), dilation=dilation),
        )

        self.non_linearity = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        conv1 = self.conv1(input)
        conv2 = self.non_linearity(self.conv2(conv1) + input)
        return conv2
