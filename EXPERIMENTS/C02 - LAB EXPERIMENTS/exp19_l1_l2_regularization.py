import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=300, n_features=10, n_informative=5,
                       noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

models = {
    "Ordinary Linear": LinearRegression(),
    "L1 (Lasso)": Lasso(alpha=2.0, max_iter=10000),
    "L2 (Ridge)": Ridge(alpha=2.0)
}

for name, estimator in models.items():
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fitted = model.named_steps[list(model.named_steps)[-1]]
    print("\n", name)
    print("Test MSE:", round(mean_squared_error(y_test, pred), 3))
    print("Coefficients:", np.round(fitted.coef_, 3))
    print("Number of near-zero coefficients:", int(np.sum(np.abs(fitted.coef_) < 1e-3)))

print("\nL1 can force some coefficients to zero; L2 generally shrinks coefficients smoothly.")
