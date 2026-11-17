import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 0]**2 + 0.15 * X[:, 0]**3
y += np.random.normal(0, 2.0, size=len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print("Degree | Train MSE | Test MSE")
print("-----------------------------")
for degree in [1, 2, 3, 5, 10, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"{degree:6d} | {train_mse:9.3f} | {test_mse:8.3f}")

print("\nLow degree may underfit (high bias); very high degree may overfit (high variance).")
