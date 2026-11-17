import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Two input values -> two hidden neurons -> one output neuron
x = np.array([[0.6], [0.1]])
W1 = np.array([[0.2, 0.4], [0.3, 0.1]])
b1 = np.array([[0.1], [0.2]])
W2 = np.array([[0.7, 0.5]])
b2 = np.array([[0.1]])

z1 = W1 @ x + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)

print("Input:\n", x)
print("\nHidden layer weighted sum (z1):\n", z1)
print("Hidden layer output (a1):\n", a1)
print("\nOutput layer weighted sum (z2):\n", z2)
print("Final network output (a2):\n", a2)
