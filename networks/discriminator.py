# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:30:17 2020

@author: ricard.deza.tripiana
"""

import torch
import torch.nn as nn
from torch.nn import init
# import functools
# from torch.optim import lr_scheduler

class Identity(nn.Module):
    def forward(self, x):
        return x

# def get_scheduler(optimizer, opt):
#     """Return a learning rate scheduler
#     Parameters:
#         optimizer          -- the optimizer of the network
#         opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
#                               opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
#     For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
#     and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
#     For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
#     See https://pytorch.org/docs/stable/optim.html for more details.
#     """
#     if opt.lr_policy == 'linear':
#         def lambda_rule(epoch):
#             lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
#             return lr_l
#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif opt.lr_policy == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
#     elif opt.lr_policy == 'plateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#     elif opt.lr_policy == 'cosine':
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
#     return scheduler

# def init_weights(net, init_type='normal', init_gain=0.02):
#     """Initialize network weights.
#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
#     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
#     work better for some applications. Feel free to try yourself.
#     """
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)  # apply the initialization function <init_func>

# def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     Parameters:
#         net (network)      -- the network to be initialized
#         init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         gain (float)       -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#     Return an initialized network.
#     """
#     if len(gpu_ids) > 0:
#         assert(torch.cuda.is_available())
#         net.to(gpu_ids[0])
#         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#     init_weights(net, init_type, init_gain=init_gain)
#     return net

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