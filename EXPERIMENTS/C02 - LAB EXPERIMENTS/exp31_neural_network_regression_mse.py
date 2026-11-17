import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X, y = make_regression(n_samples=500, n_features=4, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                 solver="adam", learning_rate_init=0.01, max_iter=1200,
                 early_stopping=True, random_state=42)
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Actual vs predicted values (first 10):")
for actual, predicted in zip(y_test[:10], pred[:10]):
    print(f"Actual: {actual:8.2f} | Predicted: {predicted:8.2f}")

print("\nTest MSE:", round(mean_squared_error(y_test, pred), 3))
print("Test MAE:", round(mean_absolute_error(y_test, pred), 3))
print("R2 score:", round(r2_score(y_test, pred), 3))
print("MSE is the average squared difference between actual and predicted values.")
