
from pathlib import Path
import numpy as np
from PIL import Image

DATASET = Path(
    r"C:\Users\sarav\OneDrive\Documents\AI - IMPLEMENTATION"
    r"\dataset\plantvillage dataset\color"
)

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "results"

# Choose one real image
extensions = {".jpg", ".jpeg", ".png", ".bmp"}
image_path = next(
    p for p in DATASET.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

with Image.open(image_path) as image:
    rgb = image.convert("RGB")
    gray = rgb.convert("L").resize((64, 64))
    pixels = np.asarray(gray, dtype=np.float32)

# Convert pixels to the [0, 1] range
normalised = pixels / 255.0

# Flatten the image
features = normalised.reshape(-1)

print("TASK 1: PREPROCESSING ANALYSIS")
print("--------------------------------")
print("Sample image:", image_path.name)
print("Original RGB dimensions:", rgb.size)
print("Original RGB array shape:", np.asarray(rgb).shape)
print("Grayscale image shape:", pixels.shape)
print("Normalised range:",
      float(normalised.min()), "to", float(normalised.max()))
print("Flattened feature shape:", features.shape)
print("Number of pixel features:", features.size)

print("\nInformation changes:")
print("- Resizing reduces spatial resolution.")
print("- Grayscale removes explicit colour-channel information.")
print("- Normalisation scales pixel values to [0, 1].")
print("- Flattening changes the representation but retains pixel values.")

OUTPUT.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT / "task01_sample_features.npy", features)

print("\nSaved sample feature vector.")
