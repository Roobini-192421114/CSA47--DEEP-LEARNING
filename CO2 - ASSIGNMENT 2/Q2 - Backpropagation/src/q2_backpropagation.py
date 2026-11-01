
import csv
from pathlib import Path

import numpy as np


# Reproducibility
SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(z):
    # Clipping prevents exponential overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y, probabilities):
    eps = 1e-12
    probabilities = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )


class NeuralNetwork:
    def __init__(self, input_size=4, hidden_size=5, seed=42):
        generator = np.random.default_rng(seed)

        # Small random initial weights
        self.W1 = generator.normal(
            0, 0.1, size=(input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = generator.normal(
            0, 0.1, size=(hidden_size, 1)
        )
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def loss(self, X, y):
        probabilities = self.forward(X)
        return binary_cross_entropy(y, probabilities)

    def backward(self, y, buggy=False):
        n = y.shape[0]

        # BCE + sigmoid output derivative
        dZ2 = (self.A2 - y) / n
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Propagate gradient to hidden layer
        dA1 = dZ2 @ self.W2.T

        if buggy:
            # Deliberate bug: omit sigmoid derivative
            dZ1 = dA1
        else:
            dZ1 = dA1 * self.A1 * (1 - self.A1)

        dW1 = self.X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

    def update(self, gradients, learning_rate=0.1):
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

    def parameters(self):
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }


def numerical_gradients(model, X, y, epsilon):
    numerical = {}

    for name, parameter in model.parameters().items():
        gradient = np.zeros_like(parameter)

        # Central finite differences
        for index in np.ndindex(parameter.shape):
            original = parameter[index]

            parameter[index] = original + epsilon
            loss_plus = model.loss(X, y)

            parameter[index] = original - epsilon
            loss_minus = model.loss(X, y)

            parameter[index] = original

            gradient[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

        numerical[name] = gradient

    # Restore cached forward-pass values
    model.forward(X)
    return numerical


def relative_error(analytical, numerical):
    numerator = np.linalg.norm(analytical - numerical)
    denominator = (
        np.linalg.norm(analytical)
        + np.linalg.norm(numerical)
        + 1e-12
    )
    return numerator / denominator


def gradient_check(model, X, y, epsilon, buggy=False):
    model.forward(X)
    analytical = model.backward(y, buggy=buggy)
    numerical = numerical_gradients(model, X, y, epsilon)

    errors = {}
    for name in analytical:
        errors[name] = relative_error(
            analytical[name], numerical[name]
        )

    return errors


def make_demo_data(samples=1000, seed=42):
    """
    Simulated hospital-style data for demonstrating the algorithm.
    Features: age, length of stay, prior admissions, medications.
    Labels are simulated and are NOT real patient outcomes.
    """
    generator = np.random.default_rng(seed)

    X = generator.normal(size=(samples, 4))

    # Give the four features interpretable approximate scales
    X[:, 0] = 50 + 15 * X[:, 0]  # age
    X[:, 1] = 5 + 3 * X[:, 1]    # length of stay
    X[:, 2] = np.maximum(0, 2 + X[:, 2])  # prior admissions
    X[:, 3] = np.maximum(0, 4 + 2 * X[:, 3])  # medications

    # Standardize for stable gradient-based learning
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mean) / std

    score = (
        0.7 * X[:, 0]
        + 0.8 * X[:, 1]
        + 1.1 * X[:, 2]
        + 0.6 * X[:, 3]
        - 0.5
    )
    probabilities = sigmoid(score)
    y = (generator.random(samples) < probabilities).astype(float)
    return X, y.reshape(-1, 1)


def train_model(X, y, epochs=200, learning_rate=0.1):
    model = NeuralNetwork(seed=SEED)
    losses = []

    for epoch in range(epochs):
        model.forward(X)
        loss = binary_cross_entropy(y, model.A2)
        gradients = model.backward(y)
        model.update(gradients, learning_rate)
        losses.append(loss)

    return model, losses


def main():
    print("=" * 60)
    print("Q2: BACKPROPAGATION AND GRADIENT CHECKING")
    print("=" * 60)

    # Small deterministic batch for gradient checking
    X_check = rng.normal(size=(8, 4))
    y_check = np.array(
        [[0.], [1.], [0.], [1.],
         [1.], [0.], [1.], [0.]]
    )

    model = NeuralNetwork(seed=SEED)

    print("\n1. CORRECT BACKPROPAGATION")
    print("Relative error acceptance target: < 1e-6")
    print(f"{'Epsilon':<12}{'Parameter':<12}{'Relative error':<18}")

    all_correct = True
    correct_results = []

    for epsilon in [1e-5, 1e-6]:
        errors = gradient_check(
            model, X_check, y_check, epsilon, buggy=False
        )

        for name, error in errors.items():
            print(f"{epsilon:<12.0e}{name:<12}{error:<18.8e}")
            correct_results.append(
                [epsilon, name, error, "correct"]
            )
            if error >= 1e-6:
                all_correct = False

    print("\nGradient check passed:", all_correct)

    print("\n2. DELIBERATE BACKPROPAGATION BUG")
    print("Bug: omitted sigmoid derivative in hidden layer.")
    print(f"{'Epsilon':<12}{'Parameter':<12}{'Relative error':<18}")

    bug_results = []
    bug_errors = {}

    for epsilon in [1e-5, 1e-6]:
        errors = gradient_check(
            model, X_check, y_check, epsilon, buggy=True
        )

        for name, error in errors.items():
            print(f"{epsilon:<12.0e}{name:<12}{error:<18.8e}")
            bug_results.append(
                [epsilon, name, error, "buggy"]
            )
            bug_errors[name] = max(
                bug_errors.get(name, 0), error
            )

    affected = [
        name for name in ["W1", "b1"]
        if bug_errors.get(name, 0) >= 1e-6
    ]
    print("First affected parameter group(s):", affected)

    print("\n3. TRAINING DEMONSTRATION")
    X, y = make_demo_data(samples=1000, seed=SEED)

    # Fixed split: 70% train, 15% validation, 15% test
    X_train, y_train = X[:700], y[:700]
    X_val, y_val = X[700:850], y[700:850]
    X_test, y_test = X[850:], y[850:]

    print("Dataset: simulated hospital-style data")
    print("Total samples:", len(X))
    print(
        f"Train: {len(X_train)}, Validation: {len(X_val)}, "
        f"Test: {len(X_test)}"
    )
    print("Features: 4; Random seed:", SEED)

    trained_model, losses = train_model(
        X_train, y_train, epochs=200, learning_rate=0.1
    )

    def evaluate(X_part, y_part):
        probabilities = trained_model.forward(X_part)
        loss = binary_cross_entropy(y_part, probabilities)
        predictions = (probabilities >= 0.5).astype(float)
        accuracy = np.mean(predictions == y_part)
        return loss, accuracy

    train_loss, train_acc = evaluate(X_train, y_train)
    val_loss, val_acc = evaluate(X_val, y_val)
    test_loss, test_acc = evaluate(X_test, y_test)

    print(f"Final train loss: {train_loss:.6f}")
    print(f"Validation loss:  {val_loss:.6f}")
    print(f"Test loss:        {test_loss:.6f}")
    print(f"Train accuracy:   {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:    {test_acc:.4f}")

    # Save gradient checking results
    csv_path = OUTPUT_DIR / "q2_gradient_check.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["epsilon", "parameter", "relative_error", "version"]
        )
        writer.writerows(correct_results)
        writer.writerows(bug_results)

    # Save training results
    metrics_path = OUTPUT_DIR / "q2_training_metrics.csv"
    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows([
            ["seed", SEED],
            ["dataset", "simulated hospital-style data"],
            ["total_samples", len(X)],
            ["train_samples", len(X_train)],
            ["validation_samples", len(X_val)],
            ["test_samples", len(X_test)],
            ["epochs", len(losses)],
            ["final_train_loss", train_loss],
            ["validation_loss", val_loss],
            ["test_loss", test_loss],
            ["train_accuracy", train_acc],
            ["validation_accuracy", val_acc],
            ["test_accuracy", test_acc],
        ])

    print("\nSaved evidence:")
    print(csv_path)
    print(metrics_path)
    print("\nQ2 script finished.")


if __name__ == "__main__":
    main()
