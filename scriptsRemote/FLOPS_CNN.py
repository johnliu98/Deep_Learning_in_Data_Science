import torch
from torchvision.models import resnet18
from shufflenet_model import ShuffleNet
from pthflops import count_ops
from conv_model import ConvNet

# Create a network and a corresponding input
device = 'cuda:0'
model = ConvNet()
inp = torch.randn(2,3,32,32)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# Count the number of FLOPs
count_ops(model, inp)
