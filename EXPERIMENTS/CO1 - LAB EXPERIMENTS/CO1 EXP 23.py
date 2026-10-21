# Question 3: Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Generate sample nonlinear data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X.ravel()**2 + np.sin(3 * X.ravel()) + np.random.randn(100) * 0.5

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models of degree 2 and 15
degrees = [2, 15]

plt.scatter(X, y, color="blue", label="Actual Data")

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("Polynomial Degree:", degree)
    print("Training MSE:", mean_squared_error(y_train, train_pred))
    print("Testing MSE:", mean_squared_error(y_test, test_pred))
    print()

    # Plot the fitted curve
    X_curve = np.linspace(-3, 3, 200).reshape(-1, 1)
    y_curve = model.predict(X_curve)

    plt.plot(X_curve, y_curve, label=f"Degree {degree}")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression and Overfitting")
plt.legend()
plt.show()
