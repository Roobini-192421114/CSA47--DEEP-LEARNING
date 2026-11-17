import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# A large network with little regularization can overfit this small dataset.
model = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu",
                      alpha=0.00001, max_iter=1500, random_state=42)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("Training accuracy:", round(train_acc, 4))
print("Testing accuracy :", round(test_acc, 4))
print("Training error   :", round(1 - train_acc, 4))
print("Testing error    :", round(1 - test_acc, 4))
print("\nA large gap (training much better than testing) indicates overfitting.")
print("Possible improvements: more data, early stopping, smaller network, L2 regularization, dropout.")
