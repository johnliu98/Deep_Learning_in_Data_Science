
import math
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import seaborn as sns

from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D

file_8 = open("./hyperparameter_tuning/ShuffleNet_g_8.pkl", "rb")
file_12 = open("./hyperparameter_tuning/ShuffleNet_g_12.pkl", "rb")
file_c = open("./hyperparameter_tuning/ConvNet.pkl", "rb")

data_8 = pickle.load(file_8)
data_12 = pickle.load(file_12)
data_c = pickle.load(file_c)

file_8.close()
file_12.close()
file_c.close()

accs_8 = []; lrs_8 = []; bss_8 = []
for (acc, params) in data_8:
    accs_8.append(acc)
    lrs_8.append(np.log(params.learning_rate))
    bss_8.append(params.batch_size)

accs_12 = []; lrs_12 = []; bss_12 = []
for (acc, params) in data_12:
    accs_12.append(acc)
    lrs_12.append(np.log(params.learning_rate))
    bss_12.append(params.batch_size)

accs_c = []; lrs_c = []; bss_c = []
for (acc, params) in data_c:
    accs_c.append(acc)
    lrs_c.append(np.log(params.learning_rate))
    bss_c.append(params.batch_size)

#create regular grid
xmin = min(lrs_8+lrs_12+lrs_c); xmax = max(lrs_8+lrs_12+lrs_c)
ymin = min(bss_8+bss_12+bss_c); ymax = max(bss_8+bss_12+bss_c)

xi, yi = np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
xi, yi = np.meshgrid(xi, yi)

#interpolate missing data
rbf8 = Rbf(lrs_8, bss_8, accs_8, function='linear')
zi8 = rbf8(xi, yi)

rbf12 = Rbf(lrs_12, bss_12, accs_12, function='linear')
zi12 = rbf12(xi, yi)


rbfc = Rbf(lrs_c, bss_c, accs_c, function='linear')
zic = rbfc(xi, yi)

xi = np.exp(xi)

#plot data
fig, ax = plt.subplots()
cs = ax.contourf(xi, yi, zi8)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('test accuracy')
ax.set_xlabel('learning rate')
ax.set_ylabel('batch size')
ax.set_xscale('log')
plt.savefig('./hyperparameter_tuning/ShuffleNet_g_8.eps', format='eps')

fig, ax = plt.subplots()
cs = ax.contourf(xi, yi, zi12)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('test accuracy')
ax.set_xlabel('learning rate')
ax.set_ylabel('batch size')
ax.set_xscale('log')
plt.savefig('./hyperparameter_tuning/ShuffleNet_g_12.eps', format='eps')

fig, ax = plt.subplots()
cs = ax.contourf(xi, yi, zic)
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel('test accuracy')
ax.set_xlabel('learning rate')
ax.set_ylabel('batch size')
ax.set_xscale('log')
plt.savefig('./hyperparameter_tuning/ConvNet.eps', format='eps')

plt.show()

data_8 = sorted(data_8, key=lambda x: x[0])
print('********HYPERPARAMETER TUNING WITH GROUPS=8********')
for (acc, params) in data_8:
    print("Accuracy: %.1f || Hyperparameters: learning_rate=%.5f, batch_size=%.0f, epochs=%.0f, gruops=%.0f" % \
    (acc, params.learning_rate, params.batch_size, params.epochs, params.groups))

data_12 = sorted(data_12, key=lambda x: x[0])
print('********HYPERPARAMETER TUNING WITH GROUPS=12********')
for (acc, params) in data_12:
    print("Accuracy: %.1f || Hyperparameters: learning_rate=%.5f, batch_size=%.0f, epochs=%.0f, gruops=%.0f" % \
    (acc, params.learning_rate, params.batch_size, params.epochs, params.groups))

data_c = sorted(data_c, key=lambda x: x[0])
print('********HYPERPARAMETER TUNING CONVNET********')
for (acc, params) in data_c:
    print("Accuracy: %.1f || Hyperparameters: learning_rate=%.5f, batch_size=%.0f, epochs=%.0f, gruops=%.0f" % \
    (acc, params.learning_rate, params.batch_size, params.epochs, params.groups))
