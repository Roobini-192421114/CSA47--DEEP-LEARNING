
import shutil
from pathlib import Path

# Current project folder
project = Path(__file__).resolve().parent

# Dataset is in the parent folder
source = project.parent / "NEU-CLS" / "NEU-CLS"

destination = project / "dataset" / "raw"

categories = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches"
}

if not source.exists():
    raise FileNotFoundError(f"Dataset not found: {source}")

counts = {name: 0 for name in categories.values()}

print("Dataset found:", source)

for image in source.rglob("*"):

    if not image.is_file():
        continue

    if image.suffix.lower() not in [".bmp", ".jpg", ".jpeg", ".png"]:
        continue

    prefix = image.stem.split("_")[0]

    if prefix in categories:
        category = categories[prefix]

        folder = destination / category
        folder.mkdir(parents=True, exist_ok=True)

        target = folder / image.name

        if not target.exists():
            shutil.copy2(image, target)

        counts[category] += 1

print("\n========== DATASET SUMMARY ==========")

for category, count in counts.items():
    print(f"{category}: {count} images")

print("------------------------------------")
print("TOTAL IMAGES:", sum(counts.values()))

if sum(counts.values()) == 1800 and all(v == 300 for v in counts.values()):
    print("SUCCESS: All six classes verified!")
else:
    print("WARNING: Check the dataset counts.")

print("Output folder:", destination)
