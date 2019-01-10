import sys
from copy import deepcopy
from time import sleep

import numpy as np

a = np.arange(8*8).reshape([8, 8])
b = np.arange(8*8, 8*8+8*8).reshape([8, 8])
print(a)
print(b)

out = np.hstack((np.reshape(a, [64]), np.reshape(b, [64])))

print(out)
