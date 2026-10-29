
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "outputs" / "results" / "task01_dataset_inventory.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "results"
GRAPH_DIR = BASE_DIR / "outputs" / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset inventory
df = pd.read_csv(DATA_FILE)

# Calculate true class probabilities
counts = df["class"].value_counts().sort_index()
p = (counts / counts.sum()).to_numpy()

# Task 3A: Entropy and class imbalance
entropy = -np.sum(p * np.log2(p))
num_classes = len(p)
max_entropy = np.log2(num_classes)
imbalance_ratio = counts.max() / counts.min()

print("\nTASK 3: PROBABILITY AND INFORMATION THEORY")
print("-" * 55)
print("Total images:", int(counts.sum()))
print("Number of classes:", num_classes)
print(f"Dataset entropy: {entropy:.4f} bits")
print(f"Maximum entropy: {max_entropy:.4f} bits")
print(f"Normalized entropy: {entropy / max_entropy:.4f}")
print(f"Largest class: {counts.idxmax()} ({counts.max()} images)")
print(f"Smallest class: {counts.idxmin()} ({counts.min()} images)")
print(f"Class imbalance ratio: {imbalance_ratio:.2f}")

# Save class probabilities and information content
results = pd.DataFrame({
    "class": counts.index,
    "image_count": counts.values,
    "probability": p,
    "information_bits": -np.log2(p)
})
results.to_csv(OUTPUT_DIR / "task03_class_probabilities.csv", index=False)

# Task 3B: Cross-entropy and KL divergence
# Illustrative untrained model: uniform prediction across all classes.
# This is a mathematical baseline, NOT a measured model prediction.
q_uniform = np.full(num_classes, 1.0 / num_classes)

cross_entropy_uniform = -np.sum(p * np.log2(q_uniform))
kl_p_uniform = np.sum(p * np.log2(p / q_uniform))

# Task 3C: Demonstrate KL divergence asymmetry with two distributions
p_example = np.array([0.8, 0.2])
q_example = np.array([0.5, 0.5])

kl_p_q = np.sum(p_example * np.log2(p_example / q_example))
kl_q_p = np.sum(q_example * np.log2(q_example / p_example))

print("\nINFORMATION-THEORY COMPARISON")
print("-" * 55)
print("Uniform prediction is an illustrative baseline.")
print(f"H(p): {entropy:.4f} bits")
print(f"H(p, uniform q): {cross_entropy_uniform:.4f} bits")
print(f"KL(p || uniform q): {kl_p_uniform:.4f} bits")
print(f"Cross-entropy - entropy: "
      f"{cross_entropy_uniform - entropy:.4f} bits")

print("\nKL ASYMMETRY EXAMPLE (illustrative distributions)")
print("p = [0.8, 0.2], q = [0.5, 0.5]")
print(f"KL(p || q): {kl_p_q:.4f} bits")
print(f"KL(q || p): {kl_q_p:.4f} bits")

# Save information-theory summary
summary = pd.DataFrame([
    {"quantity": "Dataset entropy H(p)", "value_bits": entropy,
     "status": "Measured from class counts"},
    {"quantity": "Maximum entropy", "value_bits": max_entropy,
     "status": "Theoretical maximum"},
    {"quantity": "Cross-entropy with uniform q", 
     "value_bits": cross_entropy_uniform,
     "status": "Illustrative uniform prediction"},
    {"quantity": "KL(p || uniform q)", "value_bits": kl_p_uniform,
     "status": "Illustrative uniform prediction"},
    {"quantity": "KL(p_example || q_example)", "value_bits": kl_p_q,
     "status": "Illustrative asymmetry example"},
    {"quantity": "KL(q_example || p_example)", "value_bits": kl_q_p,
     "status": "Illustrative asymmetry example"}
])
summary.to_csv(OUTPUT_DIR / "task03_information_theory_summary.csv",
               index=False)

# Plot class probabilities
plt.figure(figsize=(14, 6))
plt.bar(range(num_classes), p)
plt.xlabel("Class index")
plt.ylabel("Probability")
plt.title("Task 3: PlantVillage Class Probabilities")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "task03_class_probabilities.png", dpi=150)
plt.close()

print("\nSaved:")
print("outputs/results/task03_class_probabilities.csv")
print("outputs/results/task03_information_theory_summary.csv")
print("outputs/graphs/task03_class_probabilities.png")
print("\nTask 3 calculations completed.")
print("Note: Actual model-to-model comparisons require predicted probabilities.")
