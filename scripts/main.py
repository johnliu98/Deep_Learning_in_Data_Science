
import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader

import numpy as np

from shufflenet_model import ShuffleNet

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

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

EPOCHS = 1
BATCH_SIZE = 100
GROUPS = 1

def load_cifar10_data():
    # Upload data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)

    inds = torch.randperm(len(dataset))
    train_inds = inds[:45000]
    val_inds = inds[45000:]

    trainset = Subset(dataset, train_inds)
    valset = Subset(dataset, val_inds)
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

if __name__ == "__main__":
    trainloader, valloader, testloader = load_cifar10_data()

    net = ShuffleNet(groups=GROUPS, num_classes=len(CLASSES))
    net.backward(trainloader, valloader, epochs=EPOCHS)
    net.accuracy(testloader)

