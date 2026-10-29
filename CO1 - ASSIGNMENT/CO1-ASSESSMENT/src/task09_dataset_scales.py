
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST = BASE_DIR / "outputs" / "results" / "task01_split_manifest.csv"
RESULTS_DIR = BASE_DIR / "outputs" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (16, 16)
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

print("=" * 55)
print("TASK 9: DATASET SCALE AND MODEL EVALUATION")
print("=" * 55)

# --------------------------------------------------
# 2. Load dataset manifest
# --------------------------------------------------
print("\nLoading dataset manifest...")

df = pd.read_csv(MANIFEST)
df.columns = df.columns.str.strip()

train_df = df[
    df["split"].astype(str).str.strip() == "train"
].copy()

test_df = df[
    df["split"].astype(str).str.strip() == "test"
].copy()

print("Training images available:", len(train_df))
print("Testing images available:", len(test_df))

# --------------------------------------------------
# 3. Load images
# --------------------------------------------------
def load_images(frame, limit=None):
    if limit is not None:
        frame = frame.sample(
            n=min(limit, len(frame)),
            random_state=RANDOM_SEED
        )

    images = []
    labels = []
    failed = 0

    for i, (_, row) in enumerate(frame.iterrows(), start=1):
        try:
            with Image.open(row["image_path"]) as img:
                img = img.convert("RGB")
                img = img.resize(IMAGE_SIZE)

                arr = np.asarray(
                    img, dtype=np.float32
                ) / 255.0

                images.append(arr.flatten())
                labels.append(str(row["label"]))

        except Exception as exc:
            failed += 1
            print(f"Skipped image: {row['image_path']} ({exc})")

        if i % 1000 == 0:
            print(f"Processed {i}/{len(frame)} images")

    feature_count = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3

    if not images:
        return (
            np.empty((0, feature_count), dtype=np.float32),
            np.array([], dtype=str),
            failed
        )

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels),
        failed
    )

# --------------------------------------------------
# 4. Stable softmax
# --------------------------------------------------
def softmax(logits):
    logits = logits - np.max(
        logits, axis=1, keepdims=True
    )

    exp_values = np.exp(logits)

    return exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

# --------------------------------------------------
# 5. Train softmax regression using mini-batches
# --------------------------------------------------
def train_model(X, y, number_of_classes):
    model_rng = np.random.default_rng(RANDOM_SEED)

    # Last row is the bias parameter
    weights = np.zeros(
        (X.shape[1] + 1, number_of_classes),
        dtype=np.float32
    )

    X_bias = np.column_stack([
        X,
        np.ones(len(X), dtype=np.float32)
    ])

    for epoch in range(EPOCHS):
        indices = model_rng.permutation(len(X_bias))

        for start in range(0, len(X_bias), BATCH_SIZE):
            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            xb = X_bias[batch_indices]
            yb = y[batch_indices]

            probabilities = softmax(xb @ weights)

            targets = np.zeros_like(probabilities)
            targets[np.arange(len(yb)), yb] = 1.0

            gradient = (
                xb.T @ (probabilities - targets)
            ) / len(yb)

            weights -= LEARNING_RATE * gradient

    return weights

# --------------------------------------------------
# 6. Predict
# --------------------------------------------------
def predict_model(X, weights):
    X_bias = np.column_stack([
        X,
        np.ones(len(X), dtype=np.float32)
    ])

    probabilities = softmax(X_bias @ weights)

    return np.argmax(probabilities, axis=1)

# --------------------------------------------------
# 7. Load test images once
# --------------------------------------------------
print("\nLoading test images...")

X_test_raw, y_test_raw, test_failed = load_images(test_df)

print("Test images loaded:", len(X_test_raw))
print("Test images skipped:", test_failed)

if len(X_test_raw) == 0:
    raise RuntimeError("No test images could be loaded.")

# Fit encoder on all manifest labels for consistent class IDs
encoder = LabelEncoder()
encoder.fit(df["label"].dropna().astype(str))

y_test = encoder.transform(y_test_raw)
number_of_classes = len(encoder.classes_)

print("Number of classes:", number_of_classes)

# --------------------------------------------------
# 8. Dataset-scale experiments
# --------------------------------------------------
scales = [1000, 5000, 10000, None]

results = []

for scale in scales:
    scale_name = (
        "Full dataset" if scale is None else str(scale)
    )

    print("\n" + "=" * 55)
    print("Dataset scale:", scale_name)

    # Load training subset
    X_train_raw, y_train_raw, train_failed = load_images(
        train_df, limit=scale
    )

    if len(X_train_raw) == 0:
        print("No training images loaded. Skipping.")
        continue

    y_train = encoder.transform(y_train_raw)

    # Fit scaler using training data only
    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train_raw
    ).astype(np.float32)

    X_test = scaler.transform(
        X_test_raw
    ).astype(np.float32)

    # Train model
    print("Training softmax regression...")

    start_time = time.perf_counter()

    weights = train_model(
        X_train,
        y_train,
        number_of_classes
    )

    training_time = time.perf_counter() - start_time

    # Predict and measure inference time
    start_inference = time.perf_counter()

    predictions = predict_model(X_test, weights)

    inference_time = time.perf_counter() - start_inference

    # Evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    inference_ms_per_image = (
        inference_time / len(X_test)
    ) * 1000

    # NumPy parameter array size only
    model_size_bytes = weights.nbytes

    print("\nEvaluation results")
    print("-" * 35)
    print("Training images:", len(X_train))
    print("Training images skipped:", train_failed)
    print("Test images:", len(X_test))
    print("Test images skipped:", test_failed)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    print(
        f"Inference time per image: "
        f"{inference_ms_per_image:.4f} ms"
    )
    print(
        f"Weight memory: "
        f"{model_size_bytes / (1024 * 1024):.3f} MiB"
    )

    # Save metrics
    results.append({
        "dataset_scale": scale_name,
        "training_images_used": len(X_train),
        "training_images_skipped": train_failed,
        "test_images_used": len(X_test),
        "test_images_skipped": test_failed,
        "number_of_classes": number_of_classes,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "test_accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "training_time_seconds": training_time,
        "inference_time_total_seconds": inference_time,
        "inference_ms_per_image": inference_ms_per_image,
        "weight_memory_bytes": model_size_bytes,
        "weight_memory_MiB": model_size_bytes / (1024 * 1024)
    })

    # Save confusion matrix for this dataset scale
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(number_of_classes)
    )

    safe_name = scale_name.lower().replace(" ", "_")

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    cm_df.index.name = "actual_class"
    cm_df.columns.name = "predicted_class"

    cm_path = RESULTS_DIR / (
        f"task09_confusion_matrix_{safe_name}.csv"
    )

    cm_df.to_csv(cm_path)

    print("Confusion matrix saved:", cm_path.name)

    # Release memory before next experiment
    del X_train_raw, y_train_raw, X_train, X_test
    del weights, predictions, cm, cm_df

# --------------------------------------------------
# 9. Save final results
# --------------------------------------------------
results_df = pd.DataFrame(results)

output_file = RESULTS_DIR / "task09_dataset_scale_results.csv"

results_df.to_csv(output_file, index=False)

print("\n" + "=" * 55)
print("FINAL DATASET-SCALE RESULTS")
print("=" * 55)

if not results_df.empty:
    print(
        results_df[
            [
                "dataset_scale",
                "training_images_used",
                "test_accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "training_time_seconds",
                "inference_ms_per_image",
                "weight_memory_MiB"
            ]
        ].to_string(index=False)
    )

print("\nResults saved to:", output_file)
print("Task 9 experiment completed successfully.")
