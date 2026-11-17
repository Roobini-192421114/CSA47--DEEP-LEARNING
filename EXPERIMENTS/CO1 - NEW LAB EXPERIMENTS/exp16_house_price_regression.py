import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Features: area (sq.ft), bedrooms, location score (1 to 5)
X = np.array([
    [1000, 2, 3], [1200, 2, 4], [1500, 3, 4], [1800, 3, 5],
    [900, 1, 2], [1100, 2, 3], [1600, 3, 4], [2000, 4, 5],
    [1300, 2, 2], [1700, 3, 3], [2100, 4, 4], [1400, 3, 5],
    [950, 2, 1], [1900, 4, 4], [1550, 3, 2], [2300, 4, 5]
], dtype=float)

# Demonstration prices in lakh (synthetic values)
y = np.array([45, 55, 72, 90, 38, 50, 78, 105, 58, 80, 110, 76, 35, 98, 68, 120], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Feature order: area, bedrooms, location score")
print("Model coefficients:", np.round(model.coef_, 4))
print("Intercept:", round(model.intercept_, 4))
print("\nActual vs predicted house prices (lakh):")
for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual:6.2f} | Predicted: {predicted:6.2f}")

print("\nMAE:", round(mean_absolute_error(y_test, predictions), 3))
print("MSE:", round(mean_squared_error(y_test, predictions), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 3))
print("R2 score:", round(r2_score(y_test, predictions), 3))

# Example prediction: 1450 sq.ft, 3 bedrooms, location score 4
new_house = np.array([[1450, 3, 4]])
print("New house predicted price (lakh):", round(model.predict(new_house)[0], 2))
