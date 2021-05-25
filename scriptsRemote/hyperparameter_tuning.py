
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

file_8_1 = open("./hyperparameter_tuning/hyperparameter_tuning_8_groups_v1.pkl", "rb")
file_8_2 = open("./hyperparameter_tuning/hyperparameter_tuning_8_groups_v2.pkl", "rb")
file_12_1 = open("./hyperparameter_tuning/hyperparameter_tuning_12_groups_v1.pkl", "rb")
file_12_2 = open("./hyperparameter_tuning/hyperparameter_tuning_12_groups_v2.pkl", "rb")

data_8_1 = pickle.load(file_8_1)
data_8_2 = pickle.load(file_8_2)
data_12_1 = pickle.load(file_12_1)
data_12_2 = pickle.load(file_12_2)

file_8_1.close()
file_8_2.close()
file_12_1.close()
file_12_2.close()

data_8 = data_8_1 + data_8_2
data_12 = data_12_1 + data_12_2

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

#create regular grid
xmin = min(lrs_8+lrs_12); xmax = max(lrs_8+lrs_12)
ymin = min(bss_8+bss_12); ymax = max(bss_8+bss_12)
xi, yi = np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500)

xi, yi = np.meshgrid(xi, yi)

#interpolate missing data
rbf8 = Rbf(lrs_8, bss_8, accs_8, function='linear')
zi8 = rbf8(xi, yi)

rbf12 = Rbf(lrs_12, bss_12, accs_12, function='linear')
zi12 = rbf12(xi, yi)
xi = np.exp(xi)

#plot data
fig, ax = plt.subplots()
cs = ax.contourf(xi, yi, zi8, levels=np.linspace(30, 65, 8))
cbar = fig.colorbar(cs)
ax.set_xlabel('learning rate')
ax.set_ylabel('batch size')
ax.set_xscale('log')
plt.savefig('./hyperparameter_tuning/hyperparameter_tuning_8_groups.eps', format='eps')

fig, ax = plt.subplots()
cs = ax.contourf(xi, yi, zi12, levels=np.linspace(30, 65, 8))
cbar = fig.colorbar(cs)
ax.set_xlabel('learning rate')
ax.set_ylabel('batch size')
ax.set_xscale('log')
plt.savefig('./hyperparameter_tuning/hyperparameter_tuning_12_groups.eps', format='eps')

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
