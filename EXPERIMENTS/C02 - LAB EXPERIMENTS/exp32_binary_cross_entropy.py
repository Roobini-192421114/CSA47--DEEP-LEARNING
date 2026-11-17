import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report

X, y = make_classification(n_samples=800, n_features=10, n_informative=6,
                           n_redundant=2, weights=[0.85, 0.15],
                           random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(24, 12), activation="relu",
                  max_iter=1000, early_stopping=True, random_state=42)
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Binary cross-entropy from probabilities
eps = 1e-15
p = np.clip(probabilities, eps, 1 - eps)
bce = -np.mean(y_test * np.log(p) + (1 - y_test) * np.log(1 - p))

print("Manual binary cross-entropy:", round(bce, 5))
print("sklearn log loss:", round(log_loss(y_test, probabilities), 5))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nConfusion matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))
print("Lower binary cross-entropy indicates better-calibrated probability predictions.")
