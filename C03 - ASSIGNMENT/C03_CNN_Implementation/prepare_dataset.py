import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test folders.")
    parser.add_argument("--input", default="dataset/raw", help="Input folder with one subfolder per class.")
    parser.add_argument("--output", default="dataset/split", help="Output folder.")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy images (default). Kept for clarity.")
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-8:
        raise ValueError("Train, validation and test proportions must add up to 1.0.")

    source = Path(args.input)
    destination = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(
            f"Input folder not found: {source}. Create it and add one folder per image class."
        )

    class_dirs = sorted([p for p in source.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found inside {source}")

    rng = random.Random(args.seed)
    total = 0

    for class_dir in class_dirs:
        images = sorted([p for p in class_dir.rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
        if len(images) < 3:
            raise ValueError(
                f"Class '{class_dir.name}' has only {len(images)} images; at least 3 are required."
            )

        rng.shuffle(images)
        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        # Ensure each split gets at least one image where possible.
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"Not enough images in class '{class_dir.name}' for all splits.")

        partitions = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, split_images in partitions.items():
            target_dir = destination / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, image_path in enumerate(split_images):
                # Prefix the original relative path to avoid same-name collisions.
                safe_name = f"{index:05d}_{image_path.name}"
                shutil.copy2(image_path, target_dir / safe_name)

        total += n
        print(f"{class_dir.name}: train={len(partitions['train'])}, "
              f"val={len(partitions['val'])}, test={len(partitions['test'])}")

    print(f"\nDone. Copied {total} images into: {destination.resolve()}")
    print("Keep the original dataset unchanged; use the generated split folder for training.")

if __name__ == "__main__":
    main()
