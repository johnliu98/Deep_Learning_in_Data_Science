import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms, datasets

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

# Run on GPU if possible, otherwise run on CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU...")
else:
    device = torch.device("cpu")
    print("Running on CPU...")

# Get MNIST dataset
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):
        super(ShuffleUnit, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            groups=self.first_1x1_groups,
            batch_norm=True,
            relu=True)

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = self._make_grouped_conv3x3(
            self.bottleneck_channels,
            self.bottleneck_channels,
            groups=self.bottleneck_channels,
            padding=1,
            batch_norm=True
        )

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm=True,
            relu=False
        )

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
                              batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=1,
                         groups=groups)
        modules['conv'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def _make_grouped_conv3x3(self, in_channels, out_channels, groups,
                              padding=1, batch_norm=True):

        modules = OrderedDict()

        conv = nn.Conv2d(in_channels, out_channels,
                         kernel_size=3,
                         groups=groups,
                         padding=padding)
        modules['conv'] = conv

        if batch_norm:
            modules['batch_norm'] = batch_norm
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self):
        pass

class ShuffleNet(nn.Module):

    def __init__(self):
        super(ShuffleNet, self).__init__()

        # Define network layers here:
        # ...


    def make_data(self):
        # Extract training data
        data = None
        return data

    def convs(self, x):
        # Define the feed forward convolutional part, aka
        # the network architecture

        return x

    def forward(self, x):
        x = self.convs(x)

        # Define method for the fully-connected layer
        return x

    def test(self):
        # Compute accuracy
        accuracy = 0
        return accuracy

