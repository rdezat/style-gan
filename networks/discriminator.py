# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:30:17 2020

@author: ricard.deza.tripiana
"""

import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class DiscriminatorNet(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.lkrelu1 = nn.LeakyReLU(0.2, True)
        # n=1, nf_mult_prev=1, nf_mult=2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        self.lkrelu2 = nn.LeakyReLU(0.2, True)
        # n=2, nf_mult_prev=2, nf_mult=4       
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        self.lkrelu3 = nn.LeakyReLU(0.2, True)
        # n=3, nf_mult_prev=4, nf_mult=6
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.in4 = nn.InstanceNorm2d(512, affine=True)
        self.lkrelu4 = nn.LeakyReLU(0.2, True)
        # n_layers=3, nf_mult_prev=6, nf_mult=6
        self.zpad = nn.ZeroPad2d((1,0,1,0))
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, padding=1)

        self.output_shape = (1, 256 // 2 ** 4, 256 // 2 ** 4)

    def forward(self, X):
        """Standard forward."""
        y = self.lkrelu1(self.conv1(X))
        y = self.lkrelu2(self.in2(self.conv2(y)))
        y = self.lkrelu3(self.in3(self.conv3(y)))
        y = self.lkrelu4(self.in4(self.conv4(y)))
        y = self.zpad(y)
        y = self.conv5(y)
        return y