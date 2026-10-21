import numpy as np

X = np.array([4, 6, 8])
W = np.array([0.5, 0.3, 0.2])
b = 2

output = np.dot(X, W) + b
print("Predicted output:", output)
