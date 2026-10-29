
import numpy as np
from pathlib import Path

print("TASK 4: NUMERICAL STABILITY")
print("-" * 35)

# Experiment 1: Floating-point precision
a = np.float32(1e8)
b = np.float32(1.0)

result_float32 = (a + b) - a
result_float64 = (np.float64(1e8) + np.float64(1.0)) - np.float64(1e8)

print("\n1. Floating-point precision")
print("Float32 result:", result_float32)
print("Float64 result:", result_float64)

# Experiment 2: Unstable softmax
logits = np.array([1000.0, 1001.0, 1002.0])

with np.errstate(over="ignore", invalid="ignore"):
    exp_values = np.exp(logits)
    unstable = exp_values / np.sum(exp_values)

print("\n2. Softmax comparison")
print("Unstable softmax:", unstable)

# Numerically stable softmax
shifted_logits = logits - np.max(logits)
exp_values = np.exp(shifted_logits)
stable = exp_values / np.sum(exp_values)

print("Stable softmax:", stable)
print("Probability sum:", np.sum(stable))

# Experiment 3: Log-sum-exp
stable_log_sum_exp = (
    np.max(logits)
    + np.log(np.sum(np.exp(logits - np.max(logits))))
)

print("\n3. Stable log-sum-exp:", stable_log_sum_exp)

# Save results
output_dir = Path("outputs/results")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "task04_numerical_stability.txt", "w") as f:
    f.write(f"Float32 result: {result_float32}\n")
    f.write(f"Float64 result: {result_float64}\n")
    f.write(f"Unstable softmax: {unstable}\n")
    f.write(f"Stable softmax: {stable}\n")
    f.write(f"Probability sum: {np.sum(stable)}\n")
    f.write(f"Stable log-sum-exp: {stable_log_sum_exp}\n")

print("\nSaved: outputs/results/task04_numerical_stability.txt")
print("Task 4 experiment completed.")
