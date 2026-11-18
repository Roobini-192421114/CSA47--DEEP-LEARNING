import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from data_utils import create_dataloaders
from models import build_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_one(model_name, args, device):
    train_loader, val_loader, _, classes = create_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = build_model(model_name, len(classes), pretrained=not args.no_pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_dir = Path(args.output) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = -1.0
    history = []
    start_time = time.perf_counter()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 65}\nTraining {model_name} on {device}")
    print(f"Classes: {classes}")
    print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += batch_size

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                batch_size = labels.size(0)
                val_loss_sum += loss.item() * batch_size
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        train_accuracy = train_correct / max(train_count, 1)
        val_accuracy = val_correct / max(val_count, 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "image_size": 224,
                "pretrained": not args.no_pretrained
            }, output_dir / "best_model.pth")

    training_seconds = time.perf_counter() - start_time
    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    metadata = {
        "model": model_name,
        "classes": classes,
        "epochs_completed": len(history),
        "best_validation_accuracy": best_val_accuracy,
        "training_time_seconds": training_seconds,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device)
    }
    with open(output_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Training time: {training_seconds:.1f} seconds")
    print(f"Saved best checkpoint: {output_dir / 'best_model.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Train Custom CNN and/or MobileNetV2.")
    parser.add_argument("--model", choices=["custom", "mobilenetv2", "all"], default="all")
    parser.add_argument("--data", default="dataset/split")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ["custom", "mobilenetv2"] if args.model == "all" else [args.model]

    for name in model_names:
        # Reset seed so both experiments start reproducibly.
        set_seed(args.seed)
        run_one(name, args, device)

    print("\nTraining completed. Next run: python evaluate.py --model all --data", args.data)

if __name__ == "__main__":
    main()
