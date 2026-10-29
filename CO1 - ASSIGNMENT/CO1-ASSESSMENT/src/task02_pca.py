
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "results"
GRAPHS = ROOT / "outputs" / "graphs"

OUTPUT.mkdir(parents=True, exist_ok=True)
GRAPHS.mkdir(parents=True, exist_ok=True)

MANIFEST = OUTPUT / "task01_split_manifest.csv"

if not MANIFEST.exists():
    raise FileNotFoundError(
        "Run src/task01_split.py first."
    )

df = pd.read_csv(MANIFEST)
train = df[df["split"] == "train"]

# Select a reproducible subset of training images
sample_size = min(1000, len(train))
sample = train.sample(n=sample_size, random_state=42)

features = []

print(f"Loading {sample_size} training images...")

for i, path in enumerate(sample["image_path"], start=1):
    with Image.open(path) as image:
        gray = image.convert("L").resize((64, 64))
        pixels = np.asarray(gray, dtype=np.float32) / 255.0
        features.append(pixels.reshape(-1))

    if i % 200 == 0:
        print(f"Loaded {i}/{sample_size}")

X = np.stack(features)

print("\nTASK 2: PCA ANALYSIS")
print("--------------------")
print("Feature matrix shape:", X.shape)

# PCA is fitted only on training images
max_components = min(100, X.shape[0] - 1, X.shape[1])

pca = PCA(
    n_components=max_components,
    svd_solver="randomized",
    random_state=42
)

pca.fit(X)

ratios = pca.explained_variance_ratio_
cumulative = np.cumsum(ratios)

print("\nFirst five singular values:")
print(np.round(pca.singular_values_[:5], 4))

print("\nCumulative explained variance:")
for k in [5, 10, 20, 50, 100]:
    if k <= max_components:
        print(f"k={k}: {cumulative[k - 1] * 100:.2f}%")

# Reconstruction error for different component counts
errors = []

for k in [5, 10, 20, 50, 100]:
    if k <= max_components:
        compressed = pca.transform(X)[:, :k]
        reconstructed = (
            compressed @ pca.components_[:k]
            + pca.mean_
        )
        mse = np.mean((X - reconstructed) ** 2)
        errors.append({"components": k, "reconstruction_mse": mse})

error_df = pd.DataFrame(errors)
error_df.to_csv(
    OUTPUT / "task02_reconstruction_error.csv",
    index=False
)

spectrum_df = pd.DataFrame({
    "component": np.arange(1, max_components + 1),
    "singular_value": pca.singular_values_,
    "explained_variance_ratio": ratios,
    "cumulative_explained_variance": cumulative
})

spectrum_df.to_csv(
    OUTPUT / "task02_pca_spectrum.csv",
    index=False
)

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(1, max_components + 1),
    cumulative * 100
)
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative explained variance (%)")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig(GRAPHS / "task02_explained_variance.png", dpi=150)
plt.close()

# Plot reconstruction error
plt.figure(figsize=(7, 5))
plt.plot(
    error_df["components"],
    error_df["reconstruction_mse"],
    marker="o"
)
plt.xlabel("Number of principal components")
plt.ylabel("Reconstruction MSE")
plt.title("PCA Reconstruction Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(GRAPHS / "task02_reconstruction_error.png", dpi=150)
plt.close()

# Visualise leading principal components
fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(
        pca.components_[i].reshape(64, 64),
        cmap="gray"
    )
    ax.set_title(f"PC {i + 1}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(GRAPHS / "task02_principal_components.png", dpi=150)
plt.close()

print("\nSaved:")
print("outputs/results/task02_pca_spectrum.csv")
print("outputs/results/task02_reconstruction_error.csv")
print("outputs/graphs/task02_explained_variance.png")
print("outputs/graphs/task02_reconstruction_error.png")
print("outputs/graphs/task02_principal_components.png")
print("\nTask 2 initial experiment completed.")
