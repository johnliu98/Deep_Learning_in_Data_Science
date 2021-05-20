from time import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

class ConvNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=10):
        super(ConvNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        x = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 64)  # flattening.
        self.fc2 = nn.Linear(64, self.num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=4e-5)
        self.loss_function = nn.NLLLoss()

    def convs(self, x):
        # max pooling over 2x2
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # bc this is our output layer. No activation here.
        return F.log_softmax(x, dim=1)

    def backward(self, trainloader, valloader, epochs=1):

        train_losses = []
        val_losses = []
        for epoch in range(epochs):

            train_loss = 0.0
            for i, data in enumerate(tqdm(trainloader)):
                X_batch, y_batch = data

                self.zero_grad()

                y_pred = self(X_batch)
                loss = self.loss_function(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

                # print statistics
                train_loss += loss.item()
                if i % 10 == 9:  # print every 100 mini-batches
                    train_losses.append(train_loss / 10)

                    with torch.no_grad():
                        tot_val_loss = 0.0
                        for val_data in valloader:
                            X_val, y_val = val_data
                            N_val = len(valloader)

                            y_val_pred = self(X_val)
                            val_loss = self.loss_function(y_val_pred, y_val)
                            tot_val_loss += val_loss.item()
                        val_losses.append(tot_val_loss / N_val)

                    # print('[%d, %5d] train_loss: %.3f, val_loss: %.3f' %
                    #       (epoch + 1, i + 1, train_loss / 10, tot_val_loss / N_val))
                    train_loss = 0.0

        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.show()

    def accuracy(self, testloader):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in tqdm(testloader):
                X_test, y_test = data
                # calculate outputs by running images through the network
                y_pred = self(X_test)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(y_pred.data, 1)
                total += y_test.size(0)
                correct += (predicted == y_test).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

    def save_network(self):
        path = './networks/cifar_convnet.pth'
        path = path[:-4] + str(int(time())) + path[-4:]
        torch.save(self.state_dict(), path)

    def load_network(self, path, device):
        self.load_state_dict(torch.load(path, device))




