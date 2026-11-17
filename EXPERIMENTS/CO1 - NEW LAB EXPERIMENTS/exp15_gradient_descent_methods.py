import numpy as np

np.random.seed(7)
# Synthetic linear data: y = 3x + 2 + noise
X = np.linspace(-2, 2, 80)
y = 3 * X + 2 + np.random.normal(0, 0.3, size=len(X))

def predict(w, b, x):
    return w * x + b

def batch_gd(epochs=100, lr=0.05):
    w, b = 0.0, 0.0
    for _ in range(epochs):
        error = predict(w, b, X) - y
        dw = 2 * np.mean(error * X)
        db = 2 * np.mean(error)
        w -= lr * dw
        b -= lr * db
    return w, b

def stochastic_gd(epochs=20, lr=0.01):
    w, b = 0.0, 0.0
    for _ in range(epochs):
        for i in np.random.permutation(len(X)):
            error = (w * X[i] + b) - y[i]
            w -= lr * 2 * error * X[i]
            b -= lr * 2 * error
    return w, b

def mini_batch_gd(epochs=60, batch_size=10, lr=0.03):
    w, b = 0.0, 0.0
    for _ in range(epochs):
        indices = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            batch = indices[start:start + batch_size]
            error = predict(w, b, X[batch]) - y[batch]
            dw = 2 * np.mean(error * X[batch])
            db = 2 * np.mean(error)
            w -= lr * dw
            b -= lr * db
    return w, b

def mse(w, b):
    return np.mean((predict(w, b, X) - y) ** 2)

for name, method in [
    ("Batch GD", batch_gd),
    ("Stochastic GD", stochastic_gd),
    ("Mini-batch GD", mini_batch_gd)
]:
    w, b = method()
    print(f"{name}: weight={w:.4f}, bias={b:.4f}, final MSE={mse(w, b):.5f}")

print("\nBatch GD: stable updates but processes all samples per update.")
print("Stochastic GD: frequent noisy updates; can be faster per update.")
print("Mini-batch GD: balances update speed and stability.")
