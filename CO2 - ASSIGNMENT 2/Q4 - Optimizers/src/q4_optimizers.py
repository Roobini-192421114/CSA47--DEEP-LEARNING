
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# Q4: OPTIMIZATION FOR TRAINING DEEP MODELS
# ============================================================

SEED = 42
EPOCHS = 5
LEARNING_RATES = [0.001, 0.01, 0.1]
BATCH_SIZES = [64, 128]
TARGET_VALIDATION_LOSS = 0.80

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse the dataset already downloaded for Q1
DATA_DIR = (
    PROJECT_DIR.parent
    / "Q1 - Activation Functions"
    / "data"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


def load_data():
    train_data = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    X = train_data.data.float().unsqueeze(1) / 255.0
    y = train_data.targets.clone()

    # Fixed split from official training partition
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(X), generator=generator)
    X, y = X[indices], y[indices]

    X_train, y_train = X[:5000], y[:5000]
    X_val, y_val = X[5000:6000], y[5000:6000]

    X_test = test_data.data[:1000].float().unsqueeze(1) / 255.0
    y_test = test_data.targets[:1000].clone()

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loader(X, y, batch_size, shuffle=False):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None
    )


def create_optimizer(name, parameters, lr):
    if name == "SGD":
        return torch.optim.SGD(parameters, lr=lr)

    if name == "SGD_Momentum":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=0.9
        )

    if name == "RMSProp":
        return torch.optim.RMSprop(parameters, lr=lr)

    if name == "Adam":
        return torch.optim.Adam(parameters, lr=lr)

    raise ValueError(f"Unknown optimizer: {name}")


def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                return float("nan"), float("nan")

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def run_experiment(name, lr, batch_size, data):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    set_seed(SEED)
    model = DeepClassifier().to(DEVICE)
    optimizer = create_optimizer(name, model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(
        X_train, y_train, batch_size, shuffle=True
    )
    val_loader = make_loader(X_val, y_val, batch_size)
    test_loader = make_loader(X_test, y_test, batch_size)

    history = []
    best_val_loss = float("inf")
    best_epoch = None
    best_state = None
    diverged = False
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                diverged = True
                break

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        if diverged:
            print(
                f"{name:14s} lr={lr:<6g} batch={batch_size:<3} "
                f"epoch={epoch}: non-finite training loss"
            )
            break

        train_loss = total_loss / total_samples
        val_loss, val_acc = evaluate(model, val_loader)

        if not np.isfinite(val_loss):
            diverged = True
            break

        history.append({
            "optimizer": name,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"{name:14s} lr={lr:<6g} batch={batch_size:<3} "
            f"epoch={epoch}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    epochs_to_target = next(
        (
            row["epoch"] for row in history
            if row["val_loss"] <= TARGET_VALIDATION_LOSS
        ),
        None
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
        test_loss, test_acc = evaluate(model, test_loader)
    else:
        test_loss, test_acc = float("nan"), float("nan")

    val_losses = [row["val_loss"] for row in history]
    if len(val_losses) >= 2:
        # Mean absolute change in validation loss between epochs
        oscillation = float(np.mean(np.abs(np.diff(val_losses))))
    else:
        oscillation = float("nan")

    summary = {
        "optimizer": name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs_completed": len(history),
        "target_val_loss": TARGET_VALIDATION_LOSS,
        "epochs_to_target": epochs_to_target,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_state is not None else float("nan"),
        "test_loss_at_best_val": test_loss,
        "test_accuracy_at_best_val": test_acc,
        "val_loss_oscillation": oscillation,
        "diverged": diverged,
        "elapsed_seconds": time.time() - start
    }

    return history, summary


def main():
    set_seed()

    print("=" * 72)
    print("Q4: COMPARING OPTIMIZERS, LEARNING RATES AND BATCH SIZES")
    print("=" * 72)
    print("Device:", DEVICE)
    print("Dataset root:", DATA_DIR)
    print("Target validation loss:", TARGET_VALIDATION_LOSS)

    data = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = data

    print("Dataset: Fashion-MNIST")
    print(f"Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    print("Classes=10, image size=28x28, seed=", SEED)

    optimizer_names = ["SGD", "SGD_Momentum", "RMSProp", "Adam"]
    all_histories = []
    all_summaries = []

    total_runs = (
        len(optimizer_names)
        * len(LEARNING_RATES)
        * len(BATCH_SIZES)
    )
    run_number = 0

    for optimizer_name in optimizer_names:
        for lr in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                run_number += 1
                print(
                    f"\n[{run_number}/{total_runs}] "
                    f"{optimizer_name}, lr={lr}, batch={batch_size}"
                )

                history, summary = run_experiment(
                    optimizer_name, lr, batch_size, data
                )
                all_histories.extend(history)
                all_summaries.append(summary)

    # Save epoch-wise data
    history_path = OUTPUT_DIR / "q4_epoch_history.csv"
    with history_path.open("w", newline="") as file:
        fields = [
            "optimizer", "learning_rate", "batch_size",
            "epoch", "train_loss", "val_loss", "val_accuracy"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_histories)

    # Save summary table
    summary_path = OUTPUT_DIR / "q4_summary.csv"
    with summary_path.open("w", newline="") as file:
        fields = list(all_summaries[0].keys())
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_summaries)

    # Plot validation loss for each optimizer at each learning rate.
    # Each chart line represents a separate batch size/configuration.
    for optimizer_name in optimizer_names:
        plt.figure(figsize=(10, 6))

        for lr in LEARNING_RATES:
            for batch_size in BATCH_SIZES:
                rows = [
                    row for row in all_histories
                    if row["optimizer"] == optimizer_name
                    and row["learning_rate"] == lr
                    and row["batch_size"] == batch_size
                ]

                if rows:
                    label = f"lr={lr}, batch={batch_size}"
                    plt.plot(
                        [r["epoch"] for r in rows],
                        [r["val_loss"] for r in rows],
                        marker="o",
                        label=label
                    )

        plt.axhline(
            TARGET_VALIDATION_LOSS,
            linestyle="--",
            label=f"Target={TARGET_VALIDATION_LOSS}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Validation cross-entropy loss")
        plt.title(f"Q4: {optimizer_name} Validation Loss")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"q4_{optimizer_name.lower()}_curves.png",
            dpi=150
        )
        plt.close()

    metadata = {
        "seed": SEED,
        "dataset": "Fashion-MNIST",
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "epochs_per_run": EPOCHS,
        "optimizers": optimizer_names,
        "learning_rates": LEARNING_RATES,
        "batch_sizes": BATCH_SIZES,
        "target_validation_loss": TARGET_VALIDATION_LOSS,
        "device": str(DEVICE),
        "total_configurations": total_runs
    }

    with (OUTPUT_DIR / "q4_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print("\nSUMMARY")
    for row in all_summaries:
        print(
            f"{row['optimizer']:14s} "
            f"lr={row['learning_rate']:<6g} "
            f"batch={row['batch_size']:<3} "
            f"best_val={row['best_val_loss']:.4f} "
            f"test_acc={row['test_accuracy_at_best_val']:.3f} "
            f"target_epoch={row['epochs_to_target']} "
            f"diverged={row['diverged']}"
        )

    print("\nSaved evidence:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)

    print("\nQ4 experiments finished.")


if __name__ == "__main__":
    main()
