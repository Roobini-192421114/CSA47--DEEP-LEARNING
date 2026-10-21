import numpy as np

W = np.array([0.7, 0.5, 0.9])
G = np.array([0.2, 0.1, 0.3])
lr = 0.1

W = W - lr * G
print("Updated weights:", W)
