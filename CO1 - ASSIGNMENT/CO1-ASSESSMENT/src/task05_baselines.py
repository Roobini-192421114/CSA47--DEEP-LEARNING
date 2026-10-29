
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MANIFEST = (
    BASE_DIR / "outputs" / "results" /
    "task01_split_manifest.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_SIZE = 1000
TEST_SIZE = 500
IMAGE_SIZE = (32, 32)
PCA_COMPONENTS = 100

# --------------------------------------------------
# 2. Load the existing dataset split
# --------------------------------------------------
df = pd.read_csv(MANIFEST)

train_df = df[df["split"] == "train"].sample(
    n=TRAIN_SIZE, random_state=RANDOM_STATE
)

test_df = df[df["split"] == "test"].sample(
    n=TEST_SIZE, random_state=RANDOM_STATE
)

print("TASK 5: BASELINE ALGORITHM COMPARISON")
print("-" * 45)
print("Training images:", len(train_df))
print("Testing images:", len(test_df))
print("Image format: RGB")
print("Image size:", IMAGE_SIZE)

# --------------------------------------------------
# 3. Load RGB images and extract pixel features
# --------------------------------------------------
def load_images(frame, description):
    features = []
    labels = []

    for i, (_, row) in enumerate(
        frame.iterrows(), start=1
    ):
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            array = np.asarray(
                image, dtype=np.float32
            ) / 255.0

            features.append(array.flatten())

        labels.append(row["label"])

        if i % 100 == 0:
            print(
                f"{description}: loaded "
                f"{i}/{len(frame)} images"
            )

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)

    return X, y


print("\nLoading training images...")
X_train, y_train = load_images(
    train_df, "Training"
)

print("\nLoading testing images...")
X_test, y_test = load_images(
    test_df, "Testing"
)

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

# --------------------------------------------------
# 4. Fit PCA using training data only
# --------------------------------------------------
print("\nApplying PCA...")

pca = PCA(
    n_components=PCA_COMPONENTS,
    svd_solver="randomized",
    random_state=RANDOM_STATE
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA training shape:", X_train_pca.shape)
print(
    "Variance retained:",
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)

# --------------------------------------------------
# 5. Define three baseline algorithms
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        random_state=RANDOM_STATE
    ),

    "Linear SVM": LinearSVC(
        max_iter=10000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
}

# --------------------------------------------------
# 6. Train and evaluate all models
# --------------------------------------------------
results = []

feature_sets = [
    ("Raw RGB pixels", X_train, X_test),
    ("PCA 100 components", X_train_pca, X_test_pca)
]

for feature_name, Xtr, Xte in feature_sets:

    for model_name, model in models.items():

        print(
            f"\nTraining {model_name} "
            f"using {feature_name}..."
        )

        model.fit(Xtr, y_train)
        predictions = model.predict(Xte)

        accuracy = accuracy_score(
            y_test, predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        results.append({
            "features": feature_name,
            "algorithm": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

# --------------------------------------------------
# 7. Save and display the comparison
# --------------------------------------------------
results_df = pd.DataFrame(results)

results_path = (
    OUTPUT_DIR / "task05_baseline_results.csv"
)

results_df.to_csv(results_path, index=False)

print("\nFINAL COMPARISON")
print("=" * 75)
print(results_df.to_string(index=False))

print("\nResults saved to:")
print(results_path)

print("\nTask 5 experiment completed.")
