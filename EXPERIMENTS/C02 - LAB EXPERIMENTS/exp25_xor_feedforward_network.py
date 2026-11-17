import numpy as np
from sklearn.neural_network import MLPClassifier

# XOR is not linearly separable, so a hidden layer is required.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh",
                      solver="lbfgs", max_iter=5000, random_state=1)
model.fit(X, y)

print("XOR truth table")
for inputs, target, prediction in zip(X, y, model.predict(X)):
    print(inputs, "target =", target, "predicted =", prediction)
print("\nTraining accuracy:", model.score(X, y))
print("The hidden layer allows the network to learn the non-linear XOR boundary.")
