from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                  max_iter=2000, random_state=42, early_stopping=True)
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Iris species:", iris.target_names)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification report:\n",
      classification_report(y_test, pred, target_names=iris.target_names))
