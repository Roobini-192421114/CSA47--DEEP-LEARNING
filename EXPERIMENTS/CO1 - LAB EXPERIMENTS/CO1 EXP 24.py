# Question 4: Temperature Prediction

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Generate sample temperature data
np.random.seed(42)
X = np.linspace(0, 24, 100).reshape(-1, 1)
y = 25 + 5 * np.sin(2 * np.pi * X.ravel() / 24) + np.random.randn(100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compare polynomial degrees
degrees = [1, 3, 12]

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

    # Plot the predicted temperature curve
    X_curve = np.linspace(0, 24, 200).reshape(-1, 1)
    y_curve = model.predict(X_curve)

    plt.plot(X_curve, y_curve, label=f"Degree {degree}")

plt.scatter(X, y, color="black", s=15, label="Actual Temperature")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature")
plt.title("Temperature Prediction Using Polynomial Regression")
plt.legend()
plt.show()
