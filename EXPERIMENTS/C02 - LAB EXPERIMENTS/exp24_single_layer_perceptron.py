import numpy as np

# Perceptron learns AND logic using a single layer
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1])

weights = np.zeros(2)
bias = 0.0
learning_rate = 0.1
epochs = 20

for epoch in range(epochs):
    errors = 0
    for inputs, target in zip(X, y):
        output = 1 if np.dot(inputs, weights) + bias >= 0.5 else 0
        update = learning_rate * (target - output)
        weights += update * inputs
        bias += update
        errors += int(update != 0)
    if errors == 0:
        break

print("Final weights:", weights)
print("Final bias:", bias)
print("\nPredictions:")
for inputs, target in zip(X, y):
    output = 1 if np.dot(inputs, weights) + bias >= 0.5 else 0
    print(inputs.astype(int), "target =", target, "predicted =", output)
