
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST = BASE_DIR / "outputs" / "results" / "task01_split_manifest.csv"
RESULT_DIR = BASE_DIR / "outputs" / "results"
GRAPH_DIR = BASE_DIR / "outputs" / "graphs"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
IMAGE_SIZE = (32, 32)

# Use the same training and validation data for every model
df = pd.read_csv(MANIFEST)

train_df = df[df["split"] == "train"].sample(
    n=1000, random_state=RANDOM_STATE
).reset_index(drop=True)

val_df = df[df["split"] == "validation"].sample(
    n=300, random_state=RANDOM_STATE
).reset_index(drop=True)


def load_images(frame, name):
    X, y = [], []

    for i, (_, row) in enumerate(frame.iterrows(), start=1):
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            X.append(
                np.asarray(image, dtype=np.float32).flatten() / 255.0
            )

        y.append(row["label"])

        if i % 100 == 0:
            print(f"{name}: {i}/{len(frame)} images loaded")

    return np.asarray(X, dtype=np.float32), np.asarray(y)


print("Loading training images...")
X_train, y_train = load_images(train_df, "Training")

print("\nLoading validation images...")
X_val, y_val = load_images(val_df, "Validation")

# Vary regularisation strength to study effective model complexity.
# Smaller C = stronger regularisation.
# Larger C = weaker regularisation.
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
results = []

for C in C_values:
    print(f"\nTraining Logistic Regression with C={C}...")

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    train_error = 1 - train_acc
    val_error = 1 - val_acc
    gap = train_acc - val_acc

    results.append({
        "C": C,
        "regularisation_strength_inverse_C": 1 / C,
        "training_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "training_error": train_error,
        "validation_error": val_error,
        "accuracy_gap": gap
    })

    print(f"Training accuracy:   {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Training error:      {train_error:.4f}")
    print(f"Validation error:    {val_error:.4f}")
    print(f"Generalisation gap:  {gap:.4f}")

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "task06_capacity_results.csv",
    index=False
)

# Plot training and validation errors
plt.figure(figsize=(9, 5))
plt.semilogx(
    results_df["C"], results_df["training_error"],
    marker="o", label="Training error"
)
plt.semilogx(
    results_df["C"], results_df["validation_error"],
    marker="o", label="Validation error"
)
plt.xlabel("C (higher C = weaker regularisation)")
plt.ylabel("Error rate")
plt.title("Task 6: Capacity and Generalisation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "task06_capacity_error_curves.png",
    dpi=150
)
plt.close()

# Plot generalisation gap
plt.figure(figsize=(9, 5))
plt.semilogx(
    results_df["C"], results_df["accuracy_gap"],
    marker="o"
)
plt.xlabel("C (higher C = weaker regularisation)")
plt.ylabel("Training accuracy - validation accuracy")
plt.title("Task 6: Generalisation Gap")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "task06_generalisation_gap.png",
    dpi=150
)
plt.close()

print("\nResults saved:")
print("outputs/results/task06_capacity_results.csv")
print("outputs/graphs/task06_capacity_error_curves.png")
print("outputs/graphs/task06_generalisation_gap.png")
print("\nTask 6 capacity experiment completed.")
