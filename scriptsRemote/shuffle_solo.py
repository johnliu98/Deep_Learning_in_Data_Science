import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
from tqdm import tqdm
import pickle
from copy import deepcopy

from shufflenet_model import ShuffleNet
#from conv_model import ConvNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=100, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, metavar='LR',
                        help='learning rate')
    parser.add_argument('-g', '--groups', default=8, type=int, metavar='G',
                        help='groups')
    parser.add_argument('-e', '--epochs', default=4, type=int, metavar='E',
                        help='number of total epochs to run')
    parser.add_argument('-hiters', '--hiters', default=0, type=int, metavar='Hiters',
                        help='number of random searches')
    parser.add_argument('-hyp', '--hyper', default=0, type=int, metavar='H',
                        help='hyperparameter_search_enable')
    args = parser.parse_args()

    # Data loading code
    gpu = torch.device("cuda:0")

    if args.hyper:
        hyperparameters = []
    #config = {}
    #config['lr'] = 10**(np.random.rand(10,1)*2-4)           # range 10^[-4, -2]
    #config['batch_size'] = 2**(np.random.rand(10,1)*4 + 6)  # range 2^[6, 10]

    # for lr in config['lr']:
    #     for bs in config['batch_size']:
    #
    #         args.learning_rate = lr[0]
    #         args.batch_size = int(bs[0])
    #
    #         print('\n======================================================\n')
    #         print('learning_rate=%.5f, batch_size=%.0f, groups=%.0f' % \
    #         (args.learning_rate, args.batch_size, args.groups))
    #
    #         train_loader, val_loader, test_loader = load_cifar10_data(args)
    #         model = train(train_loader, val_loader, gpu, args)
    #         acc = accuracy(test_loader, model, gpu)
    #
    #         hyperparameters.append((acc, deepcopy(args)))

        for i in range(args.hiters):
            np.random.seed() # DIFFERENT SEEDS EVERYTIME! (note, a set seed inside
    #                                                    to make init the same)
            args.learning_rate = 10**(np.random.rand()*3-4)
            args.batch_size = int(2**(np.random.rand()*3 + 6))
            print('\n======================================================\n')
            print('learning_rate=%.5f, batch_size=%.0f, groups=%.0f' % \
            (args.learning_rate, args.batch_size, args.groups))
            print('Iteration: ' + str(i) + ', total done: ' + str(i/args.hiters))
            train_loader, val_loader, test_loader = load_cifar10_data(args)
            model = train(train_loader, val_loader, gpu, args)
            acc = accuracy(test_loader, model, gpu)
            hyperparameters.append((acc, deepcopy(args)))

        hyper_file = open("hyperparameter_tuning.pkl", "wb")
        print('Saving hyperparameters...')
        pickle.dump(hyperparameters, hyper_file)
        hyper_file.close()
    else:
        train_loader, val_loader, test_loader = load_cifar10_data(args)
        model = train(train_loader, val_loader, gpu, args)
        acc = accuracy(test_loader, model, gpu)
        print(acc)


def channel_shuffle(x, groups):
    batchsize, channels, height, width = x.data.size()

    channels_per_group = channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def load_cifar10_data(args):
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

    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size,
                           shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, drop_last=True)

    return train_loader, val_loader, test_loader

def accuracy(testloader, model, gpu):

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (X_test, y_test) in tqdm(testloader):
            X_test = X_test.cuda(non_blocking=True)
            y_test = y_test.cuda(non_blocking=True)
            # calculate outputs by running images through the network
            y_pred = model(X_test).cuda(gpu)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(y_pred.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

    acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)

    return acc

def train(train_loader, val_loader, gpu, args):
    torch.manual_seed(0)
    np.random.seed(0)
    model = ShuffleNet(groups=args.groups)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # define loss function (criterion) and optimizer
    criterion = nn.NLLLoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

    return model


if __name__ == '__main__':
    main()
