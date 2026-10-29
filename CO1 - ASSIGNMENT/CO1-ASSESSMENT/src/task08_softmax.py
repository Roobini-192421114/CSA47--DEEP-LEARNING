
# ============================================================
# TASK 8: SOFTMAX REGRESSION FROM SCRATCH
# Batch GD vs SGD vs Mini-batch GD
# Learning rates: 0.001, 0.01, 0.1
# Includes loss curves and convergence comparison
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

RANDOM_SEED = 42
IMAGE_SIZE = 16
MAX_IMAGES_PER_CLASS = 15
EPOCHS = 30
BATCH_SIZE = 32

LEARNING_RATES = [0.001, 0.01, 0.1]

METHODS = [
    "Batch GD",
    "SGD",
    "Mini-batch GD"
]

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MANIFEST_PATH = os.path.join(
    PROJECT_DIR,
    "outputs",
    "results",
    "task01_split_manifest.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "results"
)

PLOTS_DIR = os.path.join(
    PROJECT_DIR,
    "outputs",
    "plots"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------
# 2. LOAD DATASET MANIFEST
# ------------------------------------------------------------

print("Loading dataset manifest...")

if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_PATH}"
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {"image_path", "label", "split"}

if not required_columns.issubset(manifest.columns):
    raise ValueError(
        "Manifest must contain: image_path, label, split"
    )

print("Manifest columns:", list(manifest.columns))

manifest = manifest.dropna(
    subset=["image_path", "label"]
).copy()

manifest["label"] = manifest["label"].astype(str)

print("Sample columns:", list(manifest.columns))


# ------------------------------------------------------------
# 3. SAMPLE IMAGES
# ------------------------------------------------------------

sampled_groups = []

for label, group in manifest.groupby("label"):
    sample_count = min(
        MAX_IMAGES_PER_CLASS,
        len(group)
    )

    sampled_groups.append(
        group.sample(
            n=sample_count,
            random_state=RANDOM_SEED
        )
    )

sampled_df = pd.concat(
    sampled_groups,
    ignore_index=True
)

sampled_df = sampled_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("Sampled images:", len(sampled_df))
print("Number of classes:", sampled_df["label"].nunique())


# ------------------------------------------------------------
# 4. IMAGE PATH RESOLUTION
# ------------------------------------------------------------

def resolve_image_path(image_path):
    """Resolve absolute or project-relative image paths."""

    image_path = os.path.expanduser(
        str(image_path).strip()
    )

    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path

    candidates = [
        os.path.join(PROJECT_DIR, image_path),
        os.path.join(
            os.path.dirname(MANIFEST_PATH),
            image_path
        )
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Preserve the original path for the error message.
    return image_path


# ------------------------------------------------------------
# 5. LOAD AND PREPROCESS IMAGES
# ------------------------------------------------------------

print("\nLoading images...")

features = []
labels = []
failed_images = 0

for index, row in sampled_df.iterrows():

    image_path = resolve_image_path(
        row["image_path"]
    )

    image = cv2.imread(image_path)

    if image is None:
        failed_images += 1

        print(
            f"Warning: Could not load image: {image_path}"
        )
        continue

    # OpenCV loads BGR; convert to RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert pixels to floating-point values in [0, 1].
    image = image.astype(np.float32) / 255.0

    # RGB 16x16 image becomes 16*16*3 = 768 features.
    feature_vector = image.flatten()

    features.append(feature_vector)
    labels.append(row["label"])

    if (index + 1) % 100 == 0:
        print(
            f"Processed {index + 1}/{len(sampled_df)} images"
        )

if len(features) == 0:
    raise RuntimeError(
        "No images loaded. Check image_path values in the manifest."
    )

X = np.asarray(features, dtype=np.float32)
y_text = np.asarray(labels)

print("Loaded image count:", len(X))
print("Feature count per image:", X.shape[1])
print("Failed images:", failed_images)


# ------------------------------------------------------------
# 6. ENCODE LABELS
# ------------------------------------------------------------

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

number_of_classes = len(encoder.classes_)

print("Number of classes:", number_of_classes)


# ------------------------------------------------------------
# 7. TRAIN-TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of classes:", number_of_classes)


# ------------------------------------------------------------
# 8. STANDARDIZE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test = scaler.transform(
    X_test
).astype(np.float32)


# ------------------------------------------------------------
# 9. NUMERICALLY STABLE SOFTMAX
# ------------------------------------------------------------

def softmax(logits):
    """
    Convert class scores to probabilities.

    Subtracting the maximum score improves numerical stability.
    """

    shifted_logits = (
        logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_values = np.exp(shifted_logits)

    probabilities = exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )

    return probabilities


# ------------------------------------------------------------
# 10. CROSS-ENTROPY LOSS
# ------------------------------------------------------------

def cross_entropy_loss(X, y, weights):
    """
    Calculate mean multiclass cross-entropy loss.
    """

    logits = X @ weights
    probabilities = softmax(logits)

    epsilon = 1e-12

    correct_class_probabilities = probabilities[
        np.arange(len(y)),
        y
    ]

    loss = -np.mean(
        np.log(
            np.clip(
                correct_class_probabilities,
                epsilon,
                1.0
            )
        )
    )

    return float(loss)


# ------------------------------------------------------------
# 11. GRADIENT CALCULATION
# ------------------------------------------------------------

def calculate_gradient(X, y, weights):
    """
    Gradient of multiclass cross-entropy loss:

        Gradient = X.T @ (P - Y) / N
    """

    number_of_samples = X.shape[0]

    logits = X @ weights
    probabilities = softmax(logits)

    # Convert integer labels into one-hot vectors.
    one_hot = np.zeros_like(probabilities)

    one_hot[
        np.arange(number_of_samples),
        y
    ] = 1.0

    gradient = (
        X.T @ (probabilities - one_hot)
    ) / number_of_samples

    return gradient


# ------------------------------------------------------------
# 12. ACCURACY CALCULATION
# ------------------------------------------------------------

def predict(X, weights):
    logits = X @ weights
    probabilities = softmax(logits)

    return np.argmax(
        probabilities,
        axis=1
    )


def calculate_accuracy(X, y, weights):
    predictions = predict(X, weights)

    return accuracy_score(
        y,
        predictions
    )


# ------------------------------------------------------------
# 13. TRAINING ALGORITHMS
# ------------------------------------------------------------

def train_softmax(
    X,
    y,
    learning_rate,
    method,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    number_of_samples = X.shape[0]
    number_of_features = X.shape[1]
    number_of_classes = len(encoder.classes_)

    weights = np.zeros(
        (number_of_features, number_of_classes),
        dtype=np.float32
    )

    # Store full training and test losses for every epoch.
    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):

        # ---------------- BATCH GD ----------------

        if method == "Batch GD":

            gradient = calculate_gradient(
                X,
                y,
                weights
            )

            weights -= learning_rate * gradient

        # ---------------- SGD ----------------

        elif method == "SGD":

            indices = rng.permutation(
                number_of_samples
            )

            for index in indices:

                X_one = X[index:index + 1]
                y_one = y[index:index + 1]

                gradient = calculate_gradient(
                    X_one,
                    y_one,
                    weights
                )

                weights -= learning_rate * gradient

        # ---------------- MINI-BATCH GD ----------------

        elif method == "Mini-batch GD":

            indices = rng.permutation(
                number_of_samples
            )

            for start in range(
                0,
                number_of_samples,
                batch_size
            ):

                batch_indices = indices[
                    start:start + batch_size
                ]

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                gradient = calculate_gradient(
                    X_batch,
                    y_batch,
                    weights
                )

                weights -= learning_rate * gradient

        else:
            raise ValueError(
                f"Unknown training method: {method}"
            )

        # Record losses after every epoch.
        train_loss = cross_entropy_loss(
            X,
            y,
            weights
        )

        test_loss = cross_entropy_loss(
            X_test,
            y_test,
            weights
        )

        train_loss_history.append(
            train_loss
        )

        test_loss_history.append(
            test_loss
        )

    return (
        weights,
        train_loss_history,
        test_loss_history
    )


# ------------------------------------------------------------
# 14. RUN ALL EXPERIMENTS
# ------------------------------------------------------------

print("\nTraining Softmax Regression models...")

results = []
all_loss_histories = []

for method in METHODS:

    for learning_rate in LEARNING_RATES:

        print(
            f"\nMethod: {method} | "
            f"Learning rate: {learning_rate}"
        )

        start_time = time.perf_counter()

        (
            weights,
            train_loss_history,
            test_loss_history
        ) = train_softmax(
            X_train,
            y_train,
            learning_rate,
            method
        )

        training_time = (
            time.perf_counter() - start_time
        )

        train_predictions = predict(
            X_train,
            weights
        )

        test_predictions = predict(
            X_test,
            weights
        )

        train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        train_loss = train_loss_history[-1]
        test_loss = test_loss_history[-1]

        print(
            f"Train accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Test accuracy:  {test_accuracy:.4f}"
        )

        print(
            f"Train loss:     {train_loss:.4f}"
        )

        print(
            f"Test loss:      {test_loss:.4f}"
        )

        print(
            f"Training time:  {training_time:.3f} seconds"
        )

        results.append({
            "method": method,
            "learning_rate": learning_rate,
            "epochs": EPOCHS,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "training_time_seconds": training_time
        })

        # Save one record per epoch.
        for epoch_index in range(EPOCHS):

            all_loss_histories.append({
                "method": method,
                "learning_rate": learning_rate,
                "epoch": epoch_index + 1,
                "train_loss": train_loss_history[epoch_index],
                "test_loss": test_loss_history[epoch_index]
            })


# ------------------------------------------------------------
# 15. SAVE CSV RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_csv = os.path.join(
    RESULTS_DIR,
    "task08_softmax_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)

history_df = pd.DataFrame(
    all_loss_histories
)

history_csv = os.path.join(
    RESULTS_DIR,
    "task08_loss_history.csv"
)

history_df.to_csv(
    history_csv,
    index=False
)

print("\nExperiment summary:")
print(
    results_df.to_string(index=False)
)

print("\nSaved results:", results_csv)
print("Saved loss history:", history_csv)


# ------------------------------------------------------------
# 16. PLOT TRAINING AND TEST LOSS CURVES
# ------------------------------------------------------------

for method in METHODS:

    method_df = history_df[
        history_df["method"] == method
    ]

    plt.figure(figsize=(10, 6))

    for learning_rate in LEARNING_RATES:

        curve_df = method_df[
            method_df["learning_rate"] == learning_rate
        ]

        plt.plot(
            curve_df["epoch"],
            curve_df["train_loss"],
            label=f"Train LR={learning_rate}"
        )

        plt.plot(
            curve_df["epoch"],
            curve_df["test_loss"],
            linestyle="--",
            label=f"Test LR={learning_rate}"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")

    plt.title(
        f"{method}: Training and Test Loss"
    )

    plt.legend(
        fontsize=8,
        ncol=2
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_method_name = (
        method.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plot_path = os.path.join(
        PLOTS_DIR,
        f"task08_{safe_method_name}_loss_curves.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved plot:", plot_path)


# ------------------------------------------------------------
# 17. COMBINED TEST-LOSS COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(11, 7))

for method in METHODS:

    for learning_rate in LEARNING_RATES:

        curve_df = history_df[
            (history_df["method"] == method)
            & (
                history_df["learning_rate"]
                == learning_rate
            )
        ]

        plt.plot(
            curve_df["epoch"],
            curve_df["test_loss"],
            label=f"{method}, LR={learning_rate}"
        )

plt.xlabel("Epoch")
plt.ylabel("Test Cross-Entropy Loss")

plt.title(
    "Task 8: Test-Loss Comparison of Optimization Methods"
)

plt.legend(
    fontsize=8,
    ncol=2
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

combined_plot = os.path.join(
    PLOTS_DIR,
    "task08_combined_test_loss.png"
)

plt.savefig(
    combined_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved combined plot:", combined_plot)


# ------------------------------------------------------------
# 18. BEST EXPERIMENT
# ------------------------------------------------------------

best_index = results_df["test_accuracy"].idxmax()
best_result = results_df.loc[best_index]

print("\nBest experiment:")
print("Method:", best_result["method"])
print("Learning rate:", best_result["learning_rate"])
print(
    f"Test accuracy: "
    f"{best_result['test_accuracy']:.4f}"
)

print("\nTask 8 completed successfully!")
print("Results folder:", RESULTS_DIR)
print("Plots folder:", PLOTS_DIR)
