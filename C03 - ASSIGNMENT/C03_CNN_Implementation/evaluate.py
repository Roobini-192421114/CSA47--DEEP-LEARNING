import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from data_utils import create_dataloaders
from models import build_model

def evaluate_one(model_name, args, device):
    _, _, test_loader, classes = create_dataloaders(
        args.data, batch_size=args.batch_size, num_workers=args.num_workers
    )
    output_dir = Path(args.output) / model_name
    checkpoint_path = output_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train this model first with train.py."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_classes = checkpoint["classes"]
    if saved_classes != classes:
        raise ValueError("Class order in checkpoint differs from current dataset folders.")

    model = build_model(model_name, len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    prediction_rows = []
    start = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.tolist())

            for true_id, pred_id, probs in zip(labels.numpy(), predictions, probabilities):
                prediction_rows.append({
                    "actual_class": classes[int(true_id)],
                    "predicted_class": classes[int(pred_id)],
                    "confidence": float(np.max(probs))
                })

    inference_seconds = time.perf_counter() - start
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "test_images": len(y_true),
        "inference_time_seconds": inference_seconds
    }
    training_info_path = output_dir / "training_info.json"
    if training_info_path.exists():
        with open(training_info_path, encoding="utf-8") as f:
            metrics.update(json.load(f))

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{model_name} – Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    history_path = output_dir / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history["epoch"], history["train_loss"], label="Train")
        axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["train_accuracy"], label="Train")
        axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        fig.suptitle(f"{model_name} – Training Curves")
        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=160)
        plt.close(fig)

    print(f"\n{model_name} test results")
    for key in ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1_score"]:
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Saved metrics and confusion matrix to {output_dir.resolve()}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model(s) on test split.")
    parser.add_argument("--model", choices=["custom", "mobilenetv2", "all"], default="all")
    parser.add_argument("--data", default="dataset/split")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = ["custom", "mobilenetv2"] if args.model == "all" else [args.model]
    all_metrics = [evaluate_one(name, args, device) for name in names]

    if len(all_metrics) == 2:
        rows = []
        for m in all_metrics:
            rows.append({
                "Model": m["model"],
                "Accuracy": m["accuracy"],
                "Precision (weighted)": m["weighted_precision"],
                "Recall (weighted)": m["weighted_recall"],
                "F1-score (weighted)": m["weighted_f1_score"],
                "Training time (seconds)": m.get("training_time_seconds", ""),
                "Total parameters": m.get("total_parameters", ""),
                "Inference time (seconds)": m["inference_time_seconds"]
            })
        comparison_path = Path(args.output) / "comparison.csv"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(comparison_path, index=False)
        print("\nComparison saved to:", comparison_path.resolve())

if __name__ == "__main__":
    main()
