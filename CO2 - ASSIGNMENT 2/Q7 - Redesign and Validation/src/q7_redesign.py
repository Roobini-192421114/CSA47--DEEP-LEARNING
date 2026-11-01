
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ---------------- Configuration ----------------
SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
TRAIN_LIMIT = 5000
VAL_LIMIT = 1000
TEST_LIMIT = 1000

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR = (
    PROJECT_DIR.parent
    / "Q1 - Activation Functions"
    / "data"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print("Device:", DEVICE)
print("Dataset:", DATA_DIR)


# ---------------- Dataset ----------------
base_transform = transforms.ToTensor()

train_full = datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=base_transform
)
test_full = datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=base_transform
)


class TranslationAugmentedDataset(Dataset):
    """Randomly translate training images by up to 2 pixels."""
    def __init__(self, source, indices, augment=False):
        self.source = source
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.source[self.indices[index]]

        if self.augment:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)

            shifted = torch.zeros_like(image)

            source_x1 = max(0, -dx)
            source_x2 = min(28, 28 - dx)
            source_y1 = max(0, -dy)
            source_y2 = min(28, 28 - dy)

            target_x1 = max(0, dx)
            target_x2 = min(28, 28 + dx)
            target_y1 = max(0, dy)
            target_y2 = min(28, 28 + dy)

            shifted[
                :, target_y1:target_y2, target_x1:target_x2
            ] = image[
                :, source_y1:source_y2, source_x1:source_x2
            ]

            image = shifted

        return image, label


train_base = TranslationAugmentedDataset(
    train_full, range(TRAIN_LIMIT), augment=False
)
train_augmented = TranslationAugmentedDataset(
    train_full, range(TRAIN_LIMIT), augment=True
)
val_set = TranslationAugmentedDataset(
    test_full, range(VAL_LIMIT), augment=False
)
test_set = TranslationAugmentedDataset(
    test_full,
    range(VAL_LIMIT, VAL_LIMIT + TEST_LIMIT),
    augment=False
)

train_base_loader = DataLoader(
    train_base, batch_size=BATCH_SIZE, shuffle=True
)
train_aug_loader = DataLoader(
    train_augmented, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)


# ---------------- CNN architecture ----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


# ---------------- Training ----------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss_sum += criterion(
                outputs, labels
            ).item() * labels.size(0)
            correct += (
                outputs.argmax(1) == labels
            ).sum().item()
            total += labels.size(0)

    return loss_sum / total, correct / total


def train_model(model, loader, model_name):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    history = []

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0

        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader)

        history.append({
            "model": model_name,
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc
        })

        print(
            f"{model_name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss={total_loss / total:.4f} "
            f"| Val accuracy={val_acc:.4f}"
        )

    elapsed = time.perf_counter() - start
    return model, history, elapsed


# ---------------- Translation robustness test ----------------
def translate_batch(images, dx, dy):
    shifted = torch.zeros_like(images)

    source_x1 = max(0, -dx)
    source_x2 = min(28, 28 - dx)
    source_y1 = max(0, -dy)
    source_y2 = min(28, 28 - dy)

    target_x1 = max(0, dx)
    target_x2 = min(28, 28 + dx)
    target_y1 = max(0, dy)
    target_y2 = min(28, 28 + dy)

    shifted[
        :, :, target_y1:target_y2, target_x1:target_x2
    ] = images[
        :, :, source_y1:source_y2, source_x1:source_x2
    ]

    return shifted


def evaluate_shift(model, loader, dx, dy):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = translate_batch(images, dx, dy)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ---------------- Run baseline and redesigned model ----------------
experiments = [
    ("Baseline CNN", train_base_loader),
    ("Translation-Augmented CNN", train_aug_loader)
]

histories = []
summary_rows = []
translation_rows = []
saved_models = {}

for name, loader in experiments:
    print(f"\n===== {name} =====")
    torch.manual_seed(SEED)

    model = SimpleCNN()
    model, history, training_seconds = train_model(
        model, loader, name
    )
    histories.extend(history)
    saved_models[name] = model

    test_loss, clean_accuracy = evaluate(model, test_loader)

    row = {
        "model": name,
        "test_loss": test_loss,
        "clean_test_accuracy": clean_accuracy,
        "training_time_seconds": training_seconds,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
    }
    summary_rows.append(row)

    print("Clean test accuracy:", round(clean_accuracy, 4))

    for dx, dy in [
        (0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)
    ]:
        score = evaluate_shift(model, test_loader, dx, dy)

        translation_rows.append({
            "model": name,
            "shift_x": dx,
            "shift_y": dy,
            "accuracy": score
        })

        print(
            f"Shift ({dx:+d},{dy:+d}) "
            f"accuracy={score:.4f}"
        )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / (
            "baseline_cnn.pth"
            if name == "Baseline CNN"
            else "translation_augmented_cnn.pth"
        )
    )


# ---------------- Save tables ----------------
summary_df = pd.DataFrame(summary_rows)
translation_df = pd.DataFrame(translation_rows)
history_df = pd.DataFrame(histories)

summary_df.to_csv(
    OUTPUT_DIR / "q7_model_comparison.csv", index=False
)
translation_df.to_csv(
    OUTPUT_DIR / "q7_translation_results.csv", index=False
)
history_df.to_csv(
    OUTPUT_DIR / "q7_training_history.csv", index=False
)


# ---------------- Plot translation results ----------------
plt.figure(figsize=(9, 5))

for name in summary_df["model"]:
    rows = translation_df[translation_df["model"] == name]
    labels = [
        f"({int(x):+d},{int(y):+d})"
        for x, y in zip(rows["shift_x"], rows["shift_y"])
    ]
    plt.plot(
        labels, rows["accuracy"],
        marker="o", label=name
    )

plt.xlabel("Translation (x, y) in pixels")
plt.ylabel("Test accuracy")
plt.title("Q7: Robustness to Image Translation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "q7_translation_robustness.png",
    dpi=150
)
plt.close()


# ---------------- Plot training losses ----------------
plt.figure(figsize=(9, 5))

for name in history_df["model"].unique():
    rows = history_df[history_df["model"] == name]
    plt.plot(
        rows["epoch"], rows["train_loss"],
        marker="o", label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Q7: Baseline vs Redesigned Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "q7_training_loss.png", dpi=150
)
plt.close()


metadata = {
    "dataset": "Fashion-MNIST",
    "seed": SEED,
    "device": str(DEVICE),
    "training_samples": TRAIN_LIMIT,
    "validation_samples": VAL_LIMIT,
    "test_samples": TEST_LIMIT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "redesign": (
        "Random zero-filled translations up to two pixels "
        "during training"
    ),
    "limitation": (
        "One seed and a small subset; findings are exploratory. "
        "Robustness must be judged from measured results."
    )
}

with open(OUTPUT_DIR / "q7_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

readme = f"""# Q7 - Redesign and Validation

## Problem identified
Q5 showed a significant test accuracy decrease when images were
shifted by two pixels. The CNN was not fully robust to translation.

## Proposed redesign
Randomly translate training images by up to two pixels, with empty
pixels filled with zero. The architecture, training subset, optimizer,
epoch count and evaluation set remain the same for both experiments.

## Measured results
{summary_df.to_string(index=False)}

## Translation results
{translation_df.to_string(index=False)}

## Interpretation
Compare clean accuracy and shifted-image accuracy. The redesign is
useful only if the measured results show an improvement in robustness
without an unacceptable loss in clean-image accuracy.

This is a single-seed experiment on a limited subset. Additional runs
and a larger evaluation are needed for stronger conclusions.
"""
(PROJECT_DIR / "README.md").write_text(readme, encoding="utf-8")

print("\n===== Q7 FINAL COMPARISON =====")
print(summary_df.to_string(index=False))
print("\n===== TRANSLATION COMPARISON =====")
print(translation_df.to_string(index=False))
print("\nQ7 completed. Outputs saved in:", OUTPUT_DIR)
