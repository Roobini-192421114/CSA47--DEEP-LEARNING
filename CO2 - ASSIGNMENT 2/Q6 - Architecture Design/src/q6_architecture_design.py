
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------- Configuration ----------------
SEED = 42
EPOCHS = 4
BATCH_SIZE = 64
TRAIN_LIMIT = 4000
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
transform = transforms.ToTensor()

train_full = datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform
)
test_full = datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

train_set = Subset(train_full, range(TRAIN_LIMIT))
val_set = Subset(test_full, range(VAL_LIMIT))
test_set = Subset(
    test_full,
    range(VAL_LIMIT, VAL_LIMIT + TEST_LIMIT)
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)


# ---------------- CNN architecture family ----------------
class SmallCNN(nn.Module):
    """One convolution stage with 8 channels."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        return self.classifier(
            self.features(x).flatten(1)
        )


class MediumCNN(nn.Module):
    """Two convolution stages with 16 and 32 channels."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        return self.classifier(
            self.features(x).flatten(1)
        )


class DeepCNN(nn.Module):
    """Three convolution stages with increasing depth."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        return self.classifier(
            self.features(x).flatten(1)
        )


# ---------------- Evaluation ----------------
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
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

    return loss_sum / total, correct / total


def train_model(model, name):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    history = []

    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        loss_total = 0.0
        samples = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * labels.size(0)
            samples += labels.size(0)

        val_loss, val_acc = evaluate(model, val_loader)

        history.append({
            "architecture": name,
            "epoch": epoch + 1,
            "train_loss": loss_total / samples,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
        })

        print(
            f"{name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Train loss={loss_total / samples:.4f} "
            f"| Val loss={val_loss:.4f} "
            f"| Val accuracy={val_acc:.4f}"
        )

    train_seconds = time.perf_counter() - start
    return model, history, train_seconds


# ---------------- Model analysis ----------------
def parameter_count(model):
    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )


def model_file_size_mb(model, filename):
    torch.save(model.state_dict(), filename)
    return Path(filename).stat().st_size / (1024 * 1024)


def estimate_macs(model):
    """Approximate multiply-accumulate operations for one 28x28 image."""
    macs = 0
    h, w, channels = 28, 28, 1

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            out_h = (
                h + 2 * layer.padding[0]
                - layer.kernel_size[0]
            ) // layer.stride[0] + 1
            out_w = (
                w + 2 * layer.padding[1]
                - layer.kernel_size[1]
            ) // layer.stride[1] + 1

            macs += (
                out_h * out_w * layer.out_channels
                * (layer.in_channels // layer.groups)
                * layer.kernel_size[0]
                * layer.kernel_size[1]
            )
            h, w, channels = out_h, out_w, layer.out_channels

        elif isinstance(layer, nn.MaxPool2d):
            kernel = layer.kernel_size
            stride = layer.stride or kernel
            if isinstance(kernel, tuple):
                kh, kw = kernel
            else:
                kh = kw = kernel
            if isinstance(stride, tuple):
                sh, sw = stride
            else:
                sh = sw = stride
            h = (h - kh) // sh + 1
            w = (w - kw) // sw + 1

        elif isinstance(layer, nn.Linear):
            macs += layer.in_features * layer.out_features

    return macs


def receptive_field(model):
    """Track receptive field and effective stride through feature layers."""
    rf = 1
    jump = 1

    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            kernel = layer.kernel_size[0]
            stride = layer.stride[0]
            rf += (kernel - 1) * jump
            jump *= stride

        elif isinstance(layer, nn.MaxPool2d):
            kernel = layer.kernel_size
            stride = layer.stride or kernel
            if isinstance(kernel, tuple):
                kernel = kernel[0]
            if isinstance(stride, tuple):
                stride = stride[0]
            rf += (kernel - 1) * jump
            jump *= stride

    return rf


def inference_time_ms(model, repeats=50):
    model.eval()
    sample = torch.zeros(1, 1, 28, 28)

    with torch.no_grad():
        for _ in range(5):
            model(sample)

        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        elapsed = time.perf_counter() - start

    return elapsed * 1000 / repeats


# ---------------- Run architecture experiments ----------------
architectures = {
    "SmallCNN": SmallCNN,
    "MediumCNN": MediumCNN,
    "DeepCNN": DeepCNN,
}

all_history = []
results = []

for name, constructor in architectures.items():
    print(f"\n===== {name} =====")
    torch.manual_seed(SEED)

    model = constructor().to(DEVICE)
    model, history, train_seconds = train_model(model, name)
    all_history.extend(history)

    test_loss, test_acc = evaluate(model, test_loader)
    params = parameter_count(model)

    weights_path = OUTPUT_DIR / f"{name}_weights.pth"
    size_mb = model_file_size_mb(model, weights_path)
    macs = estimate_macs(model)
    rf = receptive_field(model)
    latency = inference_time_ms(model)

    result = {
        "architecture": name,
        "trainable_parameters": params,
        "estimated_MACs_per_image": macs,
        "receptive_field_pixels": rf,
        "weight_file_size_MB": size_mb,
        "training_time_seconds": train_seconds,
        "inference_time_ms_per_image": latency,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history
        ),
    }
    results.append(result)

    print("Test accuracy:", round(test_acc, 4))
    print("Parameters:", params)
    print("Estimated MACs/image:", macs)
    print("Receptive field:", rf)
    print("Weight file size (MB):", round(size_mb, 4))
    print("Inference time (ms/image):", round(latency, 4))


# ---------------- Save outputs ----------------
results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "q6_architecture_comparison.csv",
    index=False
)

history_df = pd.DataFrame(all_history)
history_df.to_csv(
    OUTPUT_DIR / "q6_training_history.csv",
    index=False
)

plt.figure(figsize=(9, 5))
for name in architectures:
    rows = history_df[
        history_df["architecture"] == name
    ]
    plt.plot(
        rows["epoch"],
        rows["validation_accuracy"],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.title("Q6: Architecture Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "q6_validation_accuracy.png",
    dpi=150
)
plt.close()

plt.figure(figsize=(9, 5))
for name in architectures:
    rows = history_df[
        history_df["architecture"] == name
    ]
    plt.plot(
        rows["epoch"],
        rows["train_loss"],
        marker="o",
        label=name
    )

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Q6: Architecture Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "q6_training_loss.png",
    dpi=150
)
plt.close()

metadata = {
    "dataset": "Fashion-MNIST",
    "seed": SEED,
    "device": str(DEVICE),
    "train_samples": TRAIN_LIMIT,
    "validation_samples": VAL_LIMIT,
    "test_samples": TEST_LIMIT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "macs_note": (
        "Approximate multiply-accumulate count for one 28x28 image; "
        "bias additions, activations and pooling comparisons are not included."
    ),
    "latency_note": (
        "Measured CPU forward-pass latency for a batch of one, averaged "
        "over repeated runs after warm-up. It is environment-dependent."
    ),
    "limitations": (
        "One training run per architecture; results are exploratory, "
        "not a three-seed statistical comparison."
    ),
}
with open(OUTPUT_DIR / "q6_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

readme = f"""# Q6 - Architecture Design

Dataset: Fashion-MNIST (28x28 grayscale, 10 classes).
Training subset: {TRAIN_LIMIT}; validation subset: {VAL_LIMIT};
test subset: {TEST_LIMIT}.
Epochs: {EPOCHS}; batch size: {BATCH_SIZE}; seed: {SEED}.

## Architecture comparison
{results_df.to_string(index=False)}

## Receptive field calculation
For a layer with kernel size k and previous effective stride j:
RF_new = RF_previous + (k - 1) * j.
The effective stride is multiplied by the layer stride.
These values describe the feature extractor's theoretical receptive field.

## MACs
MACs estimate multiply-accumulate operations for a single image.
They exclude bias additions, activations and pooling comparisons.

## Deployment boundary
This experiment uses 28x28 images and CPU inference. A deployment decision
should consider validation accuracy, latency, model file size, available
memory and the target hardware. The fastest or smallest model is not
automatically the best model.

## Outputs
- q6_architecture_comparison.csv
- q6_training_history.csv
- q6_validation_accuracy.png
- q6_training_loss.png
- q6_metadata.json
- model weight files for each architecture

Results are from one seed and four epochs per model. For stronger evidence,
repeat with at least three seeds and report mean and standard deviation.
"""
(PROJECT_DIR / "README.md").write_text(readme, encoding="utf-8")

print("\nQ6 completed. Results saved in:", OUTPUT_DIR)
print(results_df.to_string(index=False))
