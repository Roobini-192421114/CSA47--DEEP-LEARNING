import argparse
import json
import os
import random
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MLP(nn.Module):
    def __init__(self, hidden_sizes, activation_name, init_mode="standard"):
        super().__init__()
        layers = []
        dims = [784] + list(hidden_sizes)
        self.hidden_linears = nn.ModuleList()
        self.hidden_acts = nn.ModuleList()
        act_cls = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}[activation_name]
        for i in range(len(hidden_sizes)):
            linear = nn.Linear(dims[i], dims[i+1])
            self.hidden_linears.append(linear)
            self.hidden_acts.append(act_cls())
        self.output = nn.Linear(hidden_sizes[-1], 10)
        self.init_mode = init_mode
        self._initialize()

    def _initialize(self):
        for layer in self.hidden_linears:
            if self.init_mode == "poor":
                nn.init.normal_(layer.weight, mean=0.0, std=1.5)
                nn.init.zeros_(layer.bias)
            else:
                # Xavier is a reasonable common baseline for sigmoid/tanh; Kaiming for ReLU.
                if isinstance(self.hidden_acts[0], nn.ReLU):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        if self.init_mode == "poor":
            nn.init.normal_(self.output.weight, mean=0.0, std=1.5)
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for linear, act in zip(self.hidden_linears, self.hidden_acts):
            x = act(linear(x))
        return self.output(x)

def get_activation_gradients(model):
    norms = {}
    handles = []
    # Parameter gradient norm for each hidden layer's weight; it is a practical proxy
    # for layer-wise gradient magnitude and is available after backward().
    for i, layer in enumerate(model.hidden_linears, start=1):
        if layer.weight.grad is None:
            norms[f"hidden_{i}"] = float("nan")
        else:
            norms[f"hidden_{i}"] = float(layer.weight.grad.detach().norm(2).cpu())
    return norms

def dead_relu_fraction(model, x_batch):
    fractions = []
    x = x_batch.view(x_batch.size(0), -1)
    for linear, act in zip(model.hidden_linears, model.hidden_acts):
        z = linear(x)
        if isinstance(act, nn.ReLU):
            # A unit is considered inactive if it outputs zero for every example in this batch.
            a = act(z)
            fractions.append(float((a == 0).all(dim=0).float().mean().item()))
            x = a
        else:
            x = act(z)
    return float(np.mean(fractions)) if fractions else float("nan")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    losses = []
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(float(criterion(logits, y).item()))
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        ys.extend(y.cpu().numpy().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(ys, preds),
        "precision_macro": precision_score(ys, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(ys, preds, average="macro", zero_division=0),
    }

def train_one(config_name, activation, run_idx, seed, args, train_ds, val_ds, test_ds, device):
    seed_everything(seed)
    if config_name == "shallow":
        hidden = [128, 64]
        init_mode = "standard"
    elif config_name == "deep":
        hidden = [128] * 8
        init_mode = "standard"
    else:
        hidden = [128] * 8
        init_mode = "poor"

    model = MLP(hidden, activation, init_mode=init_mode).to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    first_x, first_y = next(iter(train_loader))
    first_x, first_y = first_x.to(device), first_y.to(device)
    grad_norms = {}
    # Record an initial layer-wise gradient snapshot before any optimizer update.
    model.train()
    optimizer.zero_grad()
    initial_logits = model(first_x)
    initial_loss = criterion(initial_logits, first_y)
    initial_loss.backward()
    grad_norms = get_activation_gradients(model)
    dead_fraction = dead_relu_fraction(model, first_x)
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(y)
            seen += len(y)
        train_loss = running_loss / max(seen, 1)
        val = evaluate(model, val_loader, device)
        history.append({
            "config": config_name, "activation": activation, "run": run_idx,
            "seed": seed, "epoch": epoch, "train_loss": train_loss,
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "val_precision_macro": val["precision_macro"], "val_recall_macro": val["recall_macro"],
        })
    elapsed = time.perf_counter() - start
    test = evaluate(model, test_loader, device)
    row = {
        "config": config_name, "activation": activation, "run": run_idx, "seed": seed,
        "hidden_layers": len(hidden), "hidden_units": "-".join(map(str, hidden)),
        "initial_train_loss": float(initial_loss.item()),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "val_accuracy": history[-1]["val_accuracy"],
        "val_precision_macro": history[-1]["val_precision_macro"],
        "val_recall_macro": history[-1]["val_recall_macro"],
        "test_accuracy": test["accuracy"], "test_precision_macro": test["precision_macro"],
        "test_recall_macro": test["recall_macro"], "train_seconds": elapsed,
        "dead_relu_fraction_initial": dead_fraction if activation == "relu" else np.nan,
    }
    for layer, value in grad_norms.items():
        row[f"initial_grad_norm_{layer}"] = value
    return row, history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--train-limit", type=int, default=6000)
    ap.add_argument("--val-limit", type=int, default=1000)
    ap.add_argument("--test-limit", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_full = datasets.FashionMNIST(str(data_dir), train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST(str(data_dir), train=False, download=True, transform=transform)
    if args.train_limit + args.val_limit > len(train_full) or args.test_limit > len(test_full):
        raise ValueError("Requested subset sizes exceed the official Fashion-MNIST split.")
    # Fixed subset indices; seeds vary only model initialization and minibatch order across runs.
    train_ds = Subset(train_full, range(args.train_limit))
    val_ds = Subset(train_full, range(args.train_limit, args.train_limit + args.val_limit))
    test_ds = Subset(test_full, range(args.test_limit))
    metadata = {
        "dataset": "Fashion-MNIST", "official_train_size": len(train_full), "official_test_size": len(test_full),
        "used_train_size": args.train_limit, "used_validation_size": args.val_limit, "used_test_size": args.test_limit,
        "input_shape": [1, 28, 28], "classes": 10, "device": str(device), "epochs": args.epochs,
        "runs": args.runs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "base_seed": args.seed, "seeds": [args.seed + i for i in range(args.runs)],
        "note": "Validation subset is taken from the official training partition; test subset is from official test partition."
    }
    with open(outdir / "q1_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    results, histories = [], []
    configurations = ["shallow", "deep", "poor_init"]
    activations = ["sigmoid", "tanh", "relu"]
    total = len(configurations) * len(activations) * args.runs
    count = 0
    for cfg in configurations:
        for act in activations:
            for r in range(args.runs):
                count += 1
                seed = args.seed + r
                print(f"[{count}/{total}] config={cfg}, activation={act}, run={r+1}, seed={seed}")
                try:
                    row, hist = train_one(cfg, act, r+1, seed, args, train_ds, val_ds, test_ds, device)
                    results.append(row); histories.extend(hist)
                    print(f"  val_acc={row['val_accuracy']:.4f}, test_acc={row['test_accuracy']:.4f}, "
                          f"initial_loss={row['initial_train_loss']:.4f}, seconds={row['train_seconds']:.1f}")
                except Exception as exc:
                    print(f"  FAILED: {type(exc).__name__}: {exc}")
    if not results:
        raise RuntimeError("All experiments failed; inspect the error messages above.")
    df = pd.DataFrame(results)
    hist_df = pd.DataFrame(histories)
    df.to_csv(outdir / "q1_run_results.csv", index=False)
    hist_df.to_csv(outdir / "q1_epoch_history.csv", index=False)
    group_cols = ["config", "activation"]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["run", "seed", "hidden_layers"]]
    summary = df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if str(x) != ""]).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(outdir / "q1_summary.csv", index=False)

    # Plot initial gradient norms by layer and group by activation/config.
    grad_cols = [c for c in df.columns if c.startswith("initial_grad_norm_hidden_")]
    if grad_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        for (cfg, act), grp in df.groupby(["config", "activation"]):
            vals = [grp[c].replace([np.inf, -np.inf], np.nan).mean() for c in grad_cols]
            ax.plot(range(1, len(vals)+1), vals, marker="o", label=f"{cfg}-{act}")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("Hidden layer index"); ax.set_ylabel("Initial weight-gradient L2 norm")
        ax.set_title("Q1: Initial layer-wise gradient norms (mean across runs)")
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=.3); fig.tight_layout()
        fig.savefig(outdir / "q1_gradient_norms.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for (cfg, act), grp in hist_df.groupby(["config", "activation"]):
        avg = grp.groupby("epoch")["train_loss"].mean()
        ax.plot(avg.index, avg.values, marker="o", label=f"{cfg}-{act}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax.set_title("Q1: Training loss by activation and configuration")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=.3); fig.tight_layout()
    fig.savefig(outdir / "q1_training_curves.png", dpi=160); plt.close(fig)

    relu_df = df[df["activation"] == "relu"].groupby("config")["dead_relu_fraction_initial"].mean()
    if len(relu_df):
        fig, ax = plt.subplots(figsize=(7, 4))
        relu_df.plot(kind="bar", ax=ax)
        ax.set_ylabel("Fraction of inactive ReLU units"); ax.set_title("Q1: Initial dead-ReLU fraction")
        ax.set_ylim(0, 1); fig.tight_layout()
        fig.savefig(outdir / "q1_dead_relu.png", dpi=160); plt.close(fig)

    print("\nQ1 EXPERIMENTS FINISHED")
    print(f"Device: {device}")
    print(f"Successful model runs: {len(df)} / {total}")
    print(f"Run results: {outdir / 'q1_run_results.csv'}")
    print(f"Summary: {outdir / 'q1_summary.csv'}")
    print(f"History: {outdir / 'q1_epoch_history.csv'}")
    print(f"Plots: {outdir / 'q1_gradient_norms.png'}, {outdir / 'q1_training_curves.png'}, {outdir / 'q1_dead_relu.png'}")
    print("\nMean metrics by configuration and activation:")
    print(df.groupby(["config", "activation"])[["val_accuracy", "test_accuracy", "final_train_loss", "dead_relu_fraction_initial"]].mean().round(4).to_string())

if __name__ == "__main__":
    main()
