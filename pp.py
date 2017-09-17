


import numpy as np


a = np.ones((1,2))
b = np.ones((1,2))
c = np.concatenate((a,b))
print(c.shape)

with open('valami.csv'.replace('.', '_windowed_preprocessed_data.'),'a') as f:
    print(1)