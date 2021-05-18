import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms

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

class ShuffleUnit(nn.Module):

    def __init__(self):
        super(ShuffleUnit, self).__init__()

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)


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

