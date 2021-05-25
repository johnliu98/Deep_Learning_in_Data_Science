import torch
from torchvision.models import resnet18
from shufflenet_model import ShuffleNet
from pthflops import count_ops
from conv_model import ConvNet

# Create a network and a corresponding input
device = 'cuda:0'
model = ConvNet().to(device)
inp = torch.rand(1,3,224,224).to(device)

# Count the number of FLOPs
count_ops(model, inp)
