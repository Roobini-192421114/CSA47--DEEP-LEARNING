from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, log_loss

# Synthetic customer churn-style binary classification data
X, y = make_classification(n_samples=1200, n_features=12, n_informative=7,
                           n_redundant=2, class_sep=0.8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

for activation in ["relu", "logistic", "tanh"]:
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 16), activation=activation,
                      max_iter=700, early_stopping=True, random_state=42)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    print(f"\nActivation: {activation}")
    print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Test log loss:", round(log_loss(y_test, probabilities), 4))

print("\nCompare results; performance depends on data, initialization and training settings.")
