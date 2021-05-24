import numpy as np

config = {}
config['lr'] = 10**(np.random.rand(10,1)*2-4)
config['batch_size'] = np.floor(2**(np.random.rand(10,1)*4 + 6))

print(config)
