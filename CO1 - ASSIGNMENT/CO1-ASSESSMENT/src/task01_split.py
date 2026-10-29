
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

DATASET = Path(
    r"C:\Users\sarav\OneDrive\Documents\AI - IMPLEMENTATION"
    r"\dataset\plantvillage dataset\color"
)

OUT = ROOT / "outputs" / "results"
GRAPHS = ROOT / "outputs" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)
GRAPHS.mkdir(parents=True, exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp"}
records = []

# Collect image paths and labels
for folder in sorted(DATASET.iterdir()):
    if folder.is_dir():
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                records.append({
                    "image_path": str(path),
                    "label": folder.name
                })

df = pd.DataFrame(records)

# Stratified 70% train, 15% validation, 15% test
train, remaining = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=42,
    stratify=remaining["label"]
)

train["split"] = "train"
validation["split"] = "validation"
test["split"] = "test"

manifest = pd.concat(
    [train, validation, test],
    ignore_index=True
)

manifest.to_csv(OUT / "task01_split_manifest.csv", index=False)

print("TASK 1: DATA SPLIT")
print("------------------")
print("Total:", len(manifest))
print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))

print("\nClass counts by split:")
print(pd.crosstab(manifest["label"], manifest["split"]))

# Visualise class imbalance
counts = df["label"].value_counts().sort_values()

plt.figure(figsize=(10, 12))
counts.plot(kind="barh")
plt.title("PlantVillage Class Distribution")
plt.xlabel("Number of images")
plt.ylabel("Class")
plt.tight_layout()
plt.savefig(GRAPHS / "task01_class_distribution.png", dpi=150)
plt.close()

print("\nSaved split manifest and class distribution graph.")
