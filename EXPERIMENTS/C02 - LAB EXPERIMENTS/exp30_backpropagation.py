import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# One training example, 2 inputs, 2 hidden neurons, 1 output
x = np.array([[0.5], [0.9]])
target = np.array([[1.0]])
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([[0.1], [0.1]])
W2 = np.array([[0.3, 0.2]])
b2 = np.array([[0.1]])
learning_rate = 0.5

# Forward pass
z1 = W1 @ x + b1
a1 = sigmoid(z1)
z2 = W2 @ a1 + b2
prediction = sigmoid(z2)
loss = 0.5 * (target - prediction) ** 2

# Backward pass for sigmoid + squared-error loss
delta2 = (prediction - target) * prediction * (1 - prediction)
grad_W2 = delta2 @ a1.T
grad_b2 = delta2
delta1 = (W2.T @ delta2) * a1 * (1 - a1)
grad_W1 = delta1 @ x.T
grad_b1 = delta1

print("Prediction before update:", prediction.item())
print("Loss before update:", loss.item())
print("\nOutput weight gradient:\n", grad_W2)
print("Hidden weight gradient:\n", grad_W1)

W2 -= learning_rate * grad_W2
b2 -= learning_rate * grad_b2
W1 -= learning_rate * grad_W1
b1 -= learning_rate * grad_b1

# Forward pass again to verify loss reduction
new_prediction = sigmoid(W2 @ sigmoid(W1 @ x + b1) + b2)
new_loss = 0.5 * (target - new_prediction) ** 2
print("\nPrediction after update:", new_prediction.item())
print("Loss after update:", new_loss.item())
