# Question 5: Learning Curves

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate sample regression data
np.random.seed(42)
X = np.linspace(-3, 3, 200).reshape(-1, 1)
y = X.ravel()**2 + np.sin(3 * X.ravel()) + np.random.randn(200) * 0.5

# Create two models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Degree 15": make_pipeline(
        PolynomialFeatures(degree=15),
        LinearRegression()
    )
}

# Calculate and plot learning curves
for name, model in models.items():

    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 8)
    )

    # Convert negative MSE to positive MSE
    train_error = -train_scores.mean(axis=1)
    val_error = -val_scores.mean(axis=1)

    plt.plot(sizes, train_error, marker="o", label="Training Error")
    plt.plot(sizes, val_error, marker="s", label="Validation Error")

    plt.xlabel("Number of Training Samples")
    plt.ylabel("Mean Squared Error")
    plt.title("Learning Curve: " + name)
    plt.legend()
    plt.show()

    # Print the errors
    print(name)
    print("Training Errors:", train_error)
    print("Validation Errors:", val_error)
    print()
