
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "outputs" / "results"
MANIFEST = RESULT_DIR / "task01_split_manifest.csv"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

print("TASK 7: BOOTSTRAP, BIAS-VARIANCE, MLE AND MAP")
print("=" * 60)

# --------------------------------------------------
# 2. MLE and MAP using Beta prior
# --------------------------------------------------
successes = 7
failures = 3
total = successes + failures
alpha, beta = 2, 2

mle = successes / total
map_estimate = (successes + alpha - 1) / (
    total + alpha + beta - 2
)
posterior_mean = (successes + alpha) / (
    total + alpha + beta
)

print("\nPART A: MLE AND MAP")
print(f"MLE: {mle:.4f}")
print(f"MAP: {map_estimate:.4f}")
print(f"Posterior mean: {posterior_mean:.4f}")

pd.DataFrame([{
    "successes": successes,
    "failures": failures,
    "MLE": mle,
    "MAP": map_estimate,
    "posterior_mean": posterior_mean,
    "prior_alpha": alpha,
    "prior_beta": beta
}]).to_csv(
    RESULT_DIR / "task07_mle_map_results.csv",
    index=False
)

# Show how the prior influence changes with sample size.
prior_rows = []
for n in [10, 50, 100, 500, 1000]:
    # Keep the observed success proportion at 0.7
    s = int(0.7 * n)
    f = n - s
    prior_rows.append({
        "sample_size": n,
        "MLE": s / n,
        "MAP": (s + alpha - 1) / (n + alpha + beta - 2),
        "posterior_mean": (s + alpha) / (n + alpha + beta)
    })

pd.DataFrame(prior_rows).to_csv(
    RESULT_DIR / "task07_prior_sample_size.csv",
    index=False
)

# --------------------------------------------------
# 3. Load training and validation images
# --------------------------------------------------
df = pd.read_csv(MANIFEST)

train_df = df[df["split"] == "train"].sample(
    n=1000, random_state=42
).reset_index(drop=True)

val_df = df[df["split"] == "validation"].sample(
    n=300, random_state=42
).reset_index(drop=True)


def load_images(frame, name):
    X, y = [], []

    for i, (_, row) in enumerate(frame.iterrows(), start=1):
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB").resize((32, 32))
            X.append(
                np.asarray(image, dtype=np.float32).flatten() / 255.0
            )

        y.append(row["label"])

        if i % 100 == 0:
            print(f"{name}: {i}/{len(frame)} images loaded")

    return np.asarray(X), np.asarray(y)


print("\nLoading training images...")
X_train, y_train = load_images(train_df, "Training")

print("\nLoading validation images...")
X_val, y_val = load_images(val_df, "Validation")

classes = np.unique(y_train)
N_MODELS = 10

# One-hot targets: one column per class
class_to_index = {label: i for i, label in enumerate(classes)}
Y_val = np.zeros((len(y_val), len(classes)))
for i, label in enumerate(y_val):
    Y_val[i, class_to_index[label]] = 1.0

# --------------------------------------------------
# 4. Bootstrap experiments at two capacity settings
# --------------------------------------------------
# Lower C = stronger regularisation, lower effective capacity.
# Higher C = weaker regularisation, higher effective capacity.
capacity_settings = {
    "low_capacity": 0.001,
    "high_capacity": 10.0
}

all_summaries = []
all_replicas = []

for capacity_name, C_value in capacity_settings.items():
    print(f"\nPART B: {capacity_name.upper()} (C={C_value})")
    probability_replicas = []
    accuracy_values = []

    for replica in range(N_MODELS):
        print(f"Training bootstrap model {replica + 1}/{N_MODELS}")

        # Sample with replacement
        indices = rng.integers(
            0, len(X_train), size=len(X_train)
        )
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        model = LogisticRegression(
            C=C_value,
            max_iter=2000,
            solver="lbfgs",
            random_state=replica
        )
        model.fit(X_boot, y_boot)

        # Align model probability columns to common class order
        model_prob = model.predict_proba(X_val)
        aligned_prob = np.zeros((len(X_val), len(classes)))

        for j, label in enumerate(model.classes_):
            aligned_prob[:, class_to_index[label]] = model_prob[:, j]

        prediction = classes[np.argmax(aligned_prob, axis=1)]
        accuracy = accuracy_score(y_val, prediction)

        probability_replicas.append(aligned_prob)
        accuracy_values.append(accuracy)

        all_replicas.append({
            "capacity": capacity_name,
            "C": C_value,
            "replica": replica + 1,
            "validation_accuracy": accuracy
        })

        print(f"  Validation accuracy: {accuracy:.4f}")

    P = np.asarray(probability_replicas)
    accuracies = np.asarray(accuracy_values)

    # Mean predicted probability for each validation example
    mean_P = P.mean(axis=0)

    # Probability variance across bootstrap models
    variance = P.var(axis=0, ddof=1).mean()

    # Probability-based squared error, averaged over examples/classes
    squared_error = np.mean((P - Y_val[None, :, :]) ** 2)

    # Empirical decomposition:
    # E[(prediction - target)^2] =
    # (mean prediction - target)^2 + prediction variance
    bias_squared = np.mean((mean_P - Y_val) ** 2)

    # This difference should be near zero, apart from numerical effects.
    decomposition_gap = squared_error - bias_squared - variance

    ensemble_pred = classes[np.argmax(mean_P, axis=1)]
    ensemble_accuracy = accuracy_score(y_val, ensemble_pred)

    summary = {
        "capacity": capacity_name,
        "C": C_value,
        "bootstrap_replicas": N_MODELS,
        "mean_accuracy": accuracies.mean(),
        "accuracy_std": accuracies.std(ddof=1),
        "ensemble_accuracy": ensemble_accuracy,
        "bias_squared_probability_mse": bias_squared,
        "variance_probability_mse": variance,
        "total_probability_squared_error": squared_error,
        "decomposition_gap": decomposition_gap
    }
    all_summaries.append(summary)

    print("\nCapacity summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

# --------------------------------------------------
# 5. Save all results
# --------------------------------------------------
pd.DataFrame(all_replicas).to_csv(
    RESULT_DIR / "task07_capacity_bootstrap_replicas.csv",
    index=False
)

pd.DataFrame(all_summaries).to_csv(
    RESULT_DIR / "task07_bias_variance_summary.csv",
    index=False
)

print("\nSaved:")
print("outputs/results/task07_mle_map_results.csv")
print("outputs/results/task07_prior_sample_size.csv")
print("outputs/results/task07_capacity_bootstrap_replicas.csv")
print("outputs/results/task07_bias_variance_summary.csv")

print("\nTask 7 completed.")
print(
    "Note: this is a probability-based squared-error decomposition "
    "on the fixed validation set. It does not separately estimate "
    "irreducible label noise."
)
