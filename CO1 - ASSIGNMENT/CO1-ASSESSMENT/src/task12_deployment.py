
import time
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# TASK 12: NOISY LABELS AND DEPLOYMENT CONSTRAINTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST = BASE_DIR / "outputs" / "results" / "task01_split_manifest.csv"
RESULTS_DIR = BASE_DIR / "outputs" / "results"
MODELS_DIR = BASE_DIR / "outputs" / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (16, 16)
TRAIN_LIMIT = 10000
NOISE_RATE = 0.20
RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 10

rng = np.random.default_rng(RANDOM_SEED)

print("=" * 60)
print("TASK 12: LABEL NOISE AND DEPLOYMENT EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD MANIFEST
# ------------------------------------------------------------

df = pd.read_csv(MANIFEST)
df.columns = df.columns.str.strip()

train_df = df[
    df["split"].astype(str).str.strip() == "train"
].copy()

test_df = df[
    df["split"].astype(str).str.strip() == "test"
].copy()

print("Available training images:", len(train_df))
print("Available test images:", len(test_df))


# ------------------------------------------------------------
# 2. LOAD IMAGES
# ------------------------------------------------------------

def load_images(frame):
    images = []
    labels = []

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
            print("Skipped:", row["image_path"], exc)

        if i % 1000 == 0:
            print(f"Loaded {i}/{len(frame)} images")

    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=str)
    )


train_df = train_df.sample(
    n=min(TRAIN_LIMIT, len(train_df)),
    random_state=RANDOM_SEED
)

print("\nLoading training images...")
X_train, y_train_text = load_images(train_df)

print("\nLoading test images...")
X_test, y_test_text = load_images(test_df)

if len(X_train) == 0 or len(X_test) == 0:
    raise RuntimeError("No training or test images loaded.")

encoder = LabelEncoder()
encoder.fit(
    np.concatenate([y_train_text, y_test_text])
)

y_train_clean = encoder.transform(y_train_text)
y_test = encoder.transform(y_test_text)

print("\nClasses:", len(encoder.classes_))
print("Training samples loaded:", len(X_train))
print("Test samples loaded:", len(X_test))


# ------------------------------------------------------------
# 3. SCALE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)


# ------------------------------------------------------------
# 4. INTRODUCE 20% TRAINING LABEL NOISE
# ------------------------------------------------------------

y_train_noisy = y_train_clean.copy()

noise_count = int(NOISE_RATE * len(y_train_noisy))
noise_indices = rng.choice(
    len(y_train_noisy),
    size=noise_count,
    replace=False
)

number_of_classes = len(encoder.classes_)

for index in noise_indices:
    current_label = y_train_noisy[index]

    # Select a different class, ensuring the label changes.
    alternative = rng.integers(number_of_classes - 1)

    if alternative >= current_label:
        alternative += 1

    y_train_noisy[index] = alternative

actual_noise_rate = np.mean(
    y_train_noisy != y_train_clean
)

print("\nRequested label noise:", NOISE_RATE)
print("Actual changed-label proportion:", round(actual_noise_rate, 4))


# ------------------------------------------------------------
# 5. TRAIN MODEL ON NOISY LABELS
# ------------------------------------------------------------

print("\nTraining classifier...")

model = SGDClassifier(
    loss="log_loss",
    alpha=0.0001,
    max_iter=EPOCHS,
    tol=None,
    random_state=RANDOM_SEED,
    learning_rate="constant",
    eta0=LEARNING_RATE,
    average=False
)

start = time.perf_counter()

model.fit(
    X_train,
    y_train_noisy
)

training_seconds = time.perf_counter() - start

print("Training time:", round(training_seconds, 3), "seconds")


# ------------------------------------------------------------
# 6. EVALUATE ON CLEAN TEST LABELS
# ------------------------------------------------------------

print("\nEvaluating on clean test labels...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

print("Test accuracy:", round(accuracy, 4))
print("Macro F1-score:", round(macro_f1, 4))


# ------------------------------------------------------------
# 7. MEASURE INFERENCE LATENCY
# ------------------------------------------------------------

warmup_count = min(100, len(X_test))

for i in range(warmup_count):
    model.predict(X_test[i:i + 1])

timing_count = min(1000, len(X_test))

start = time.perf_counter()

for i in range(timing_count):
    model.predict(X_test[i:i + 1])

elapsed_seconds = time.perf_counter() - start

inference_ms = (
    elapsed_seconds / timing_count
) * 1000

print(
    "Desktop inference latency per image:",
    round(inference_ms, 4),
    "ms"
)


# ------------------------------------------------------------
# 8. SAVE DEPLOYABLE MODEL
# ------------------------------------------------------------

model_path = MODELS_DIR / "task12_noisy_softmax.joblib"

artifact = {
    "model": model,
    "scaler": scaler,
    "label_encoder": encoder,
    "image_size": IMAGE_SIZE
}

joblib.dump(
    artifact,
    model_path,
    compress=3
)

model_size_bytes = model_path.stat().st_size
model_size_mb = model_size_bytes / (1024 * 1024)

print("\nSaved model:", model_path)
print("Serialized model size:", round(model_size_mb, 4), "MiB")


# ------------------------------------------------------------
# 9. CHECK CONSTRAINTS AND SAVE RESULTS
# ------------------------------------------------------------

results = {
    "training_images": len(X_train),
    "test_images": len(X_test),
    "number_of_classes": number_of_classes,
    "requested_label_noise": NOISE_RATE,
    "actual_label_noise": actual_noise_rate,
    "test_accuracy": accuracy,
    "macro_f1": macro_f1,
    "training_time_seconds": training_seconds,
    "desktop_inference_ms_per_image": inference_ms,
    "serialized_model_size_bytes": model_size_bytes,
    "serialized_model_size_MiB": model_size_mb,
    "desktop_latency_below_50ms": inference_ms < 50,
    "serialized_size_below_5MB": model_size_bytes < 5_000_000,
    "smartphone_latency_verified": False,
    "smartphone_size_verified": False
}

results_path = RESULTS_DIR / "task12_deployment_results.csv"

pd.DataFrame([results]).to_csv(
    results_path,
    index=False
)

json_path = RESULTS_DIR / "task12_deployment_results.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\n" + "=" * 60)
print("TASK 12 SUMMARY")
print("=" * 60)
print(pd.DataFrame([results]).to_string(index=False))

print("\nSaved results:", results_path)
print("Saved JSON:", json_path)
print("\nTask 12 experiment completed.")
print(
    "Note: Desktop timing does not verify smartphone performance."
)
