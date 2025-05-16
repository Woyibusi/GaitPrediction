import numpy as np
y = np.load("./data/y.npy")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
