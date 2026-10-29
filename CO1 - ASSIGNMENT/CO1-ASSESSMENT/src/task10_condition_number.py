
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# TASK 10: CONDITION NUMBER ANALYSIS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST = BASE_DIR / "outputs" / "results" / "task01_split_manifest.csv"
RESULT_DIR = BASE_DIR / "outputs" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (16, 16)
SAMPLE_SIZE = 1000
RANDOM_SEED = 42

print("=" * 55)
print("TASK 10: COVARIANCE CONDITION NUMBER")
print("=" * 55)

# 1. Load training manifest
df = pd.read_csv(MANIFEST)
train_df = df[df["split"].astype(str).str.strip() == "train"]

sample_df = train_df.sample(
    n=min(SAMPLE_SIZE, len(train_df)),
    random_state=RANDOM_SEED
).reset_index(drop=True)

# 2. Load and flatten images
images = []

print("\nLoading images...")

for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
    try:
        with Image.open(row["image_path"]) as img:
            img = img.convert("RGB").resize(IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.float64) / 255.0
            images.append(arr.flatten())
    except Exception as exc:
        print("Skipping image:", exc)

    if i % 100 == 0:
        print(f"Processed {i}/{len(sample_df)} images")

X = np.asarray(images, dtype=np.float64)

if X.shape[0] < 2:
    raise RuntimeError("Not enough images to calculate covariance.")

print("\nImages loaded:", X.shape[0])
print("Features per image:", X.shape[1])

# 3. Center the data
X_centered = X - X.mean(axis=0)

# 4. Calculate covariance matrix
covariance = np.cov(X_centered, rowvar=False)

# 5. Calculate eigenvalues
eigenvalues = np.linalg.eigvalsh(covariance)
eigenvalues = np.maximum(eigenvalues, 0.0)

largest = float(eigenvalues.max())
smallest = float(eigenvalues.min())

# Ignore near-zero eigenvalues when reporting effective conditioning
tolerance = np.finfo(np.float64).eps * max(
    covariance.shape
) * largest

positive_eigenvalues = eigenvalues[eigenvalues > tolerance]

if len(positive_eigenvalues) == 0:
    condition_number = float("inf")
    effective_condition_number = float("inf")
else:
    smallest_positive = float(positive_eigenvalues.min())

    condition_number = (
        largest / smallest if smallest > 0 else float("inf")
    )

    effective_condition_number = (
        largest / smallest_positive
    )

print("\nRESULTS")
print("-" * 40)
print(f"Largest eigenvalue: {largest:.10e}")
print(f"Smallest eigenvalue: {smallest:.10e}")
print(f"Numerical tolerance: {tolerance:.10e}")
print(f"Covariance condition number: {condition_number:.6e}")
print(
    "Effective condition number (positive eigenvalues): "
    f"{effective_condition_number:.6e}"
)

# 6. Save results
result = pd.DataFrame([{
    "images_used": X.shape[0],
    "features_per_image": X.shape[1],
    "largest_eigenvalue": largest,
    "smallest_eigenvalue": smallest,
    "numerical_tolerance": tolerance,
    "covariance_condition_number": condition_number,
    "effective_condition_number": effective_condition_number
}])

output_file = RESULT_DIR / "task10_condition_number.csv"
result.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Condition-number experiment completed.")
