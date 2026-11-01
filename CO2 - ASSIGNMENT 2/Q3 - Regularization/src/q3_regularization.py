
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
# Q3: REGULARIZATION AND OVERFITTING
# ============================================================

SEED = 42
EPOCHS = 12
BATCH_SIZE = 128

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


class FashionMLP(nn.Module):
    def __init__(self, dropout=0.0, large=False):
        super().__init__()

        if large:
            hidden1, hidden2 = 512, 256
        else:
            hidden1, hidden2 = 128, 64

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 10)
        )

    def forward(self, x):
        return self.network(x)


def load_data():
    """Load the existing local Fashion-MNIST files without downloading."""
    train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )

    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    X_train = train_dataset.data.float().unsqueeze(1) / 255.0
    y_train = train_dataset.targets.clone()

    X_test = test_dataset.data.float().unsqueeze(1) / 255.0
    y_test = test_dataset.targets.clone()

    # Fixed, reproducible shuffle of the official training partition
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(X_train), generator=generator)

    X_train = X_train[indices]
    y_train = y_train[indices]

    # 5,000 training samples, 1,000 validation samples
    X_fit, y_fit = X_train[:5000], y_train[:5000]
    X_val, y_val = X_train[5000:6000], y_train[5000:6000]

    # 1,000 held-out test samples from the official test partition
    X_test, y_test = X_test[:1000], y_test[:1000]

    return X_fit, y_fit, X_val, y_val, X_test, y_test


def make_loader(X, y, shuffle=False):
    return DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


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

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def weight_norm(model):
    return float(torch.sqrt(sum(
        torch.sum(parameter.detach() ** 2)
        for name, parameter in model.named_parameters()
        if "weight" in name
    )).item())


def train_experiment(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    dropout=0.0,
    weight_decay=0.0,
    large=False,
    early_stopping=False,
    augment=False,
    epochs=EPOCHS
):
    set_seed(SEED)

    model = FashionMLP(dropout=dropout, large=large).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 3
    no_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Simple random translation augmentation
            if augment:
                shifts = torch.randint(
                    -2, 3, (images.size(0), 2), device=DEVICE
                )
                shifted = torch.zeros_like(images)

                for i in range(images.size(0)):
                    dy, dx = shifts[i].tolist()
                    shifted[i] = torch.roll(
                        images[i], shifts=(dy, dx), dims=(-2, -1)
                    )

                images = shifted

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        history.append({
            "experiment": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "accuracy_gap": train_acc - val_acc,
            "weight_norm": weight_norm(model)
        })

        print(
            f"{name:20s} | Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1

        if early_stopping and no_improvement >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Evaluate the best validation checkpoint, not necessarily the last epoch
    model.load_state_dict(best_state)
    model.to(DEVICE)

    test_loss, test_acc = evaluate(model, test_loader)

    elapsed = time.time() - started
    final_row = history[-1]

    summary = {
        "experiment": name,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
        "final_train_accuracy": final_row["train_accuracy"],
        "final_val_accuracy": final_row["val_accuracy"],
        "final_accuracy_gap": final_row["accuracy_gap"],
        "best_model_test_loss": test_loss,
        "best_model_test_accuracy": test_acc,
        "weight_norm": final_row["weight_norm"],
        "seconds": elapsed
    }

    return history, summary


def main():
    set_seed()

    print("=" * 68)
    print("Q3: ENGINEER OVERFITTING, THEN CONTROL IT")
    print("=" * 68)
    print("Device:", DEVICE)
    print("Dataset path:", DATA_DIR)

    X, y, X_val, y_val, X_test, y_test = load_data()

    print("Dataset: Fashion-MNIST")
    print("Training samples available:", len(X))
    print("Validation samples:", len(X_val))
    print("Test samples:", len(X_test))
    print("Input image shape:", tuple(X[0].shape))
    print("Classes: 10")
    print("Seed:", SEED)

    all_histories = []
    all_summaries = []

    # --------------------------------------------------------
    # Condition 1: generalisation baseline
    # Smaller model and more training data
    # --------------------------------------------------------
    print("\nCONDITION 1: GENERALISATION BASELINE")

    history, summary = train_experiment(
        name="general_baseline",
        X_train=X,
        y_train=y,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        large=False,
        epochs=EPOCHS
    )
    all_histories.extend(history)
    all_summaries.append(summary)

    # --------------------------------------------------------
    # Condition 2: deliberately encourage overfitting
    # Small training set and large model
    # --------------------------------------------------------
    print("\nCONDITION 2: OVERFITTING BASELINE")

    small_X = X[:500]
    small_y = y[:500]

    history, summary = train_experiment(
        name="overfit_baseline",
        X_train=small_X,
        y_train=small_y,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        large=True,
        epochs=EPOCHS
    )
    all_histories.extend(history)
    all_summaries.append(summary)

    # --------------------------------------------------------
    # Condition 3: label noise as an additional challenge
    # --------------------------------------------------------
    print("\nCONDITION 3: LABEL NOISE")

    noisy_y = small_y.clone()
    generator = torch.Generator().manual_seed(SEED)
    noise_indices = torch.randperm(
        len(noisy_y), generator=generator
    )[:int(0.20 * len(noisy_y))]

    for index in noise_indices:
        original = int(noisy_y[index])
        replacement = int(torch.randint(0, 9, (1,)).item())
        if replacement >= original:
            replacement += 1
        noisy_y[index] = replacement

    history, summary = train_experiment(
        name="label_noise_20pct",
        X_train=small_X,
        y_train=noisy_y,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        large=True,
        epochs=EPOCHS
    )
    all_histories.extend(history)
    all_summaries.append(summary)

    # --------------------------------------------------------
    # Compare regularisation techniques under the same
    # overfitting condition
    # --------------------------------------------------------
    print("\nREGULARISATION COMPARISON")

    variants = [
        {
            "name": "overfit_l2",
            "weight_decay": 0.01
        },
        {
            "name": "overfit_dropout",
            "dropout": 0.5
        },
        {
            "name": "overfit_early_stop",
            "early_stopping": True
        },
        {
            "name": "overfit_augmentation",
            "augment": True
        },
        {
            "name": "overfit_dropout_l2",
            "dropout": 0.3,
            "weight_decay": 0.001
        }
    ]

    for variant in variants:
        name = variant.pop("name")

        history, summary = train_experiment(
            name=name,
            X_train=small_X,
            y_train=small_y,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            large=True,
            epochs=EPOCHS,
            **variant
        )

        all_histories.extend(history)
        all_summaries.append(summary)

    # Save epoch-level evidence
    history_path = OUTPUT_DIR / "q3_epoch_history.csv"
    with history_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_histories[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_histories)

    # Save experiment summaries
    summary_path = OUTPUT_DIR / "q3_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(all_summaries[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_summaries)

    # Plot training and validation loss curves
    plt.figure(figsize=(12, 7))
    for name in dict.fromkeys(row["experiment"] for row in all_histories):
        rows = [r for r in all_histories if r["experiment"] == name]
        plt.plot(
            [r["epoch"] for r in rows],
            [r["train_loss"] for r in rows],
            label=f"{name} train"
        )
        plt.plot(
            [r["epoch"] for r in rows],
            [r["val_loss"] for r in rows],
            linestyle="--",
            label=f"{name} validation"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Q3: Training and Validation Loss")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q3_loss_curves.png", dpi=160)
    plt.close()

    # Plot train-validation accuracy gap
    plt.figure(figsize=(10, 6))
    for name in dict.fromkeys(row["experiment"] for row in all_histories):
        rows = [r for r in all_histories if r["experiment"] == name]
        plt.plot(
            [r["epoch"] for r in rows],
            [r["accuracy_gap"] for r in rows],
            label=name
        )

    plt.axhline(0, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy - validation accuracy")
    plt.title("Q3: Generalisation Gap")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q3_accuracy_gap.png", dpi=160)
    plt.close()

    metadata = {
        "seed": SEED,
        "dataset": "Fashion-MNIST",
        "dataset_source": str(DATA_DIR),
        "train_samples_baseline": len(X),
        "train_samples_overfit": len(small_X),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "classes": 10,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "regularization_methods": [
            "L2 weight decay",
            "Dropout",
            "Early stopping",
            "Data augmentation",
            "Dropout plus L2"
        ]
    }

    with (OUTPUT_DIR / "q3_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    print("\nSUMMARY")
    for row in all_summaries:
        print(
            f"{row['experiment']:22s} | "
            f"best_epoch={row['best_epoch']:2d} | "
            f"val_acc={row['final_val_accuracy']:.3f} | "
            f"test_acc={row['best_model_test_accuracy']:.3f} | "
            f"gap={row['final_accuracy_gap']:.3f}"
        )

    print("\nSaved evidence:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)

    print("\nQ3 experiments finished.")


if __name__ == "__main__":
    main()
