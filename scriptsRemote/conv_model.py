from time import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

class ConvNet(nn.Module):

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self, in_channels=3, num_classes=10):
        super(ConvNet, self).__init__()
        self.channel_sizes = [24, 90, 250, 600]
        self.layers = nn.Sequential(
            nn.Conv2d(
                3, self.channel_sizes[0],
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.channel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(
                self.channel_sizes[0], self.channel_sizes[1],
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.channel_sizes[1]),
            nn.ReLU(),
            nn.Conv2d(
                self.channel_sizes[1], self.channel_sizes[2],
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.channel_sizes[2]),
            nn.ReLU(),
            nn.Conv2d(
                self.channel_sizes[2], self.channel_sizes[3],
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(self.channel_sizes[3]),
            nn.ReLU(),
            #nn.AvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(self.channel_sizes[3], 10)
        )

        self.init_params()

    def forward(self, x):
        x = self.layers(x)
        x = F.log_softmax(x, dim=1)
        return x
