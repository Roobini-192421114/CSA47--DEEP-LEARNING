
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -------------------- Configuration --------------------
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
print("Dataset directory:", DATA_DIR)


# -------------------- Dataset --------------------
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


# -------------------- Model 1: Fully Connected --------------------
class FullyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# -------------------- Model 2: CNN --------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        return self.fc(x)


# -------------------- Training and evaluation --------------------
def accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_model(model, model_name):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = []
    start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / seen
        val_accuracy = accuracy(model, val_loader)

        history.append({
            "model": model_name,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_accuracy": val_accuracy
        })

        print(
            f"{model_name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {train_loss:.4f} "
            f"| Validation accuracy: {val_accuracy:.4f}"
        )

    training_seconds = time.perf_counter() - start_time
    return model, history, training_seconds


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# -------------------- Convolution calculations --------------------
def conv_output_size(n, kernel, stride, padding):
    return (n + 2 * padding - kernel) // stride + 1


input_size = 28
kernel_size = 3
stride = 1
padding = 1

conv_size = conv_output_size(
    input_size, kernel_size, stride, padding
)
pool_size = conv_output_size(conv_size, 2, 2, 0)

# FC first-layer parameters: weights + biases
fc_first_layer_params = 28 * 28 * 128 + 128

# CNN first convolution: 16 filters, 1 channel, 3x3 kernels + biases
cnn_first_layer_params = 16 * (1 * 3 * 3) + 16

print("\n--- Hand calculations ---")
print(
    f"Convolution output: "
    f"({28} + 2*{padding} - {kernel_size})/{stride} + 1 "
    f"= {conv_size} x {conv_size}"
)
print(f"After 2x2 pooling: {pool_size} x {pool_size}")
print("FC first-layer parameters:", fc_first_layer_params)
print("CNN first-layer parameters:", cnn_first_layer_params)


# -------------------- Train both models --------------------
fc_model, fc_history, fc_seconds = train_model(
    FullyConnectedNet(), "Fully Connected"
)

cnn_model, cnn_history, cnn_seconds = train_model(
    SimpleCNN(), "CNN"
)

fc_test_accuracy = accuracy(fc_model, test_loader)
cnn_test_accuracy = accuracy(cnn_model, test_loader)

print("\n--- Test results ---")
print(f"Fully connected test accuracy: {fc_test_accuracy:.4f}")
print(f"CNN test accuracy: {cnn_test_accuracy:.4f}")
print(f"Fully connected training time: {fc_seconds:.2f} seconds")
print(f"CNN training time: {cnn_seconds:.2f} seconds")
print("Fully connected total parameters:", count_parameters(fc_model))
print("CNN total parameters:", count_parameters(cnn_model))


# -------------------- Translation robustness --------------------
def translate_images(images, shift_x, shift_y):
    """Translate images without wrapping pixels across image edges."""
    shifted = torch.zeros_like(images)
    height, width = images.shape[-2:]

    source_x1 = max(0, -shift_x)
    source_x2 = min(width, width - shift_x)
    source_y1 = max(0, -shift_y)
    source_y2 = min(height, height - shift_y)

    target_x1 = max(0, shift_x)
    target_x2 = min(width, width + shift_x)
    target_y1 = max(0, shift_y)
    target_y2 = min(height, height + shift_y)

    shifted[
        :, :,
        target_y1:target_y2,
        target_x1:target_x2
    ] = images[
        :, :,
        source_y1:source_y2,
        source_x1:source_x2
    ]

    return shifted


def shifted_accuracy(model, loader, shift_x, shift_y):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = translate_images(images, shift_x, shift_y)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


translation_results = []

for name, model in [
    ("Fully Connected", fc_model),
    ("CNN", cnn_model)
]:
    for dx, dy in [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)]:
        score = shifted_accuracy(
            model, test_loader, dx, dy
        )
        translation_results.append({
            "model": name,
            "shift_x": dx,
            "shift_y": dy,
            "test_accuracy": score
        })
        print(
            f"{name} shift ({dx:+d},{dy:+d}): "
            f"accuracy={score:.4f}"
        )


# -------------------- Save results --------------------
history = fc_history + cnn_history
pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "q5_training_history.csv", index=False
)

pd.DataFrame(translation_results).to_csv(
    OUTPUT_DIR / "q5_translation_results.csv", index=False
)

summary = pd.DataFrame([
    {
        "model": "Fully Connected",
        "total_parameters": count_parameters(fc_model),
        "first_layer_parameters": fc_first_layer_params,
        "test_accuracy": fc_test_accuracy,
        "training_seconds": fc_seconds
    },
    {
        "model": "CNN",
        "total_parameters": count_parameters(cnn_model),
        "first_layer_parameters": cnn_first_layer_params,
        "test_accuracy": cnn_test_accuracy,
        "training_seconds": cnn_seconds
    }
])

summary.to_csv(OUTPUT_DIR / "q5_model_comparison.csv", index=False)

metadata = {
    "seed": SEED,
    "dataset": "Fashion-MNIST",
    "train_samples": TRAIN_LIMIT,
    "validation_samples": VAL_LIMIT,
    "test_samples": TEST_LIMIT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "device": str(DEVICE),
    "input_shape": [1, 28, 28],
    "conv1_output_shape": [16, conv_size, conv_size],
    "pool_output_shape": [16, pool_size, pool_size],
    "conv2_output_shape": [32, pool_size, pool_size]
}

with open(OUTPUT_DIR / "q5_metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)


# -------------------- Visualize training curves --------------------
plt.figure(figsize=(9, 5))
for model_name in ["Fully Connected", "CNN"]:
    rows = [r for r in history if r["model"] == model_name]
    plt.plot(
        [r["epoch"] for r in rows],
        [r["train_loss"] for r in rows],
        marker="o",
        label=model_name
    )

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Q5: Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q5_training_curves.png", dpi=150)
plt.close()


# -------------------- Visualize CNN filters --------------------
filters = cnn_model.conv1.weight.detach().cpu()
fig, axes = plt.subplots(4, 4, figsize=(7, 7))

for index, axis in enumerate(axes.flat):
    axis.imshow(filters[index, 0], cmap="gray")
    axis.set_title(f"Filter {index + 1}")
    axis.axis("off")

plt.suptitle("Learned First-Layer CNN Filters")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q5_learned_filters.png", dpi=150)
plt.close()


# -------------------- Visualize feature maps --------------------
sample_image, sample_label = test_full[0]
cnn_model.eval()

with torch.no_grad():
    first_features = cnn_model.relu1(
        cnn_model.conv1(sample_image.unsqueeze(0).to(DEVICE))
    ).cpu()[0]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))

for index, axis in enumerate(axes.flat):
    axis.imshow(first_features[index], cmap="viridis")
    axis.set_title(f"Map {index + 1}")
    axis.axis("off")

plt.suptitle(
    f"Q5: CNN Feature Maps (Fashion-MNIST label {sample_label})"
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q5_feature_maps.png", dpi=150)
plt.close()


# -------------------- README --------------------
readme = f"""# Q5 - CNN Design

## Dataset and setup
- Dataset: Fashion-MNIST, 28x28 grayscale images, 10 classes.
- Training subset: {TRAIN_LIMIT} images.
- Validation subset: {VAL_LIMIT} images.
- Test subset: {TEST_LIMIT} images.
- Epochs: {EPOCHS}; batch size: {BATCH_SIZE}; random seed: {SEED}.
- Device: {DEVICE}.

## Convolution output calculation
Output size = floor((N + 2P - K) / S) + 1.

For N=28, P=1, K=3, S=1:
(28 + 2 - 3) / 1 + 1 = 28.
Therefore the first convolution produces 28x28 feature maps.

2x2 max pooling with stride 2 produces 14x14 maps.

## First-layer parameter counts
Fully connected layer: 784*128 + 128 = {fc_first_layer_params}.
CNN convolution: 16*(1*3*3) + 16 = {cnn_first_layer_params}.

## Measured comparison
{summary.to_string(index=False)}

## Translation robustness
See q5_translation_results.csv for the accuracy of both models under
horizontal and vertical shifts of two pixels.

## Figures
- q5_training_curves.png
- q5_learned_filters.png
- q5_feature_maps.png

## Interpretation
Convolution uses the same filter weights at every spatial position,
which greatly reduces first-layer parameters and encourages detection
of local patterns. Pooling reduces spatial resolution and provides some
tolerance to small shifts, but does not guarantee complete translation
invariance. The actual performance conclusions must be based on the
measured CSV outputs.
"""

( PROJECT_DIR / "README.md" ).write_text(readme, encoding="utf-8")

print("\nQ5 completed. Results saved to:", OUTPUT_DIR)
print("\nFiles:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print("-", path.name)
