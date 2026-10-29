
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Update this path if your dataset is stored elsewhere
DATASET = Path(
    r"C:\Users\sarav\OneDrive\Documents\AI - IMPLEMENTATION"
    r"\dataset\plantvillage dataset\color"
)

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Find class folders and image files
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
classes = sorted(p for p in DATASET.iterdir() if p.is_dir())

rows = []
for class_folder in classes:
    for image_path in class_folder.rglob("*"):
        if image_path.suffix.lower() not in extensions:
            continue

        try:
            with Image.open(image_path) as image:
                rows.append({
                    "image": str(image_path),
                    "class": class_folder.name,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode
                })
        except Exception:
            print("Skipped unreadable image:", image_path)

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError(
        f"No images found. Check dataset path: {DATASET}"
    )

# Save dataset inventory
df.to_csv(OUTPUT / "task01_dataset_inventory.csv", index=False)

print("\nTASK 1: DATASET ANALYSIS")
print("-----------------------")
print("Dataset path:", DATASET)
print("Valid images:", len(df))
print("Number of classes:", df["class"].nunique())
print("\nImages per class:")
print(df["class"].value_counts().sort_index())
print("\nImage dimensions:")
print(df[["width", "height"]].describe())
print("\nImage colour modes:")
print(df["mode"].value_counts())

# Illustrative feature vector for norm calculations
x = np.array([3, -4, 12], dtype=float)

print("\nIllustrative vector:", x)
print("L1 norm:", np.linalg.norm(x, ord=1))
print("L2 norm:", np.linalg.norm(x, ord=2))
print("L-infinity norm:", np.linalg.norm(x, ord=np.inf))

# Example shapes after preprocessing to grayscale 64 x 64
print("\nExpected representation shapes:")
print("One image: (1, 64, 64)")
print("Batch of 32 images: (32, 1, 64, 64)")
print("Batch labels: (32,)")
print("Flattened feature vector: (4096,)")
print("Full feature matrix: (number_of_images, 4096)")

print("\nInventory saved to:", OUTPUT)
