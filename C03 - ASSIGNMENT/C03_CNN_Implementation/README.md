# CSA47 A3 – CO3: Comparative CNN Implementation

This project implements and compares:
1. **Custom CNN** – a small convolutional neural network built from scratch.
2. **MobileNetV2** – an ImageNet-pretrained transfer-learning model.

It is designed for the assignment scenario: multi-class product/surface-defect image classification.

## Dataset

Recommended dataset: **NEU Surface Defect Database (NEU-CLS)**. It contains six steel-surface defect categories:
- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

Download the dataset from its official/research-hosted source or the source specified by your faculty. This ZIP does not include the image dataset because the image files are large and have their own usage terms.

### Expected input folder

Place your downloaded images under `dataset/raw/`, with one folder per class:

```text
dataset/
└── raw/
    ├── crazing/
    ├── inclusion/
    ├── patches/
    ├── pitted_surface/
    ├── rolled-in_scale/
    └── scratches/
```

The folder names can be different; each immediate subfolder under `raw` is treated as a class. Images can be `.jpg`, `.jpeg`, `.png`, or `.bmp`.

## Setup in VS Code (Windows)

Open this extracted folder in VS Code, then open **Terminal → New Terminal**.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in that terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Step 1 – Split the dataset

Run once after placing images in `dataset/raw/`:

```powershell
python prepare_dataset.py --input dataset/raw --output dataset/split
```

It creates stratified `train`, `val`, and `test` folders (70%, 15%, 15%). Each class needs enough images for all three splits. The script copies images; it does not modify the originals.

```text
dataset/split/
├── train/class_name/
├── val/class_name/
└── test/class_name/
```

## Step 2 – Train both CNN models

```powershell
python train.py --model all --data dataset/split --epochs 15 --batch-size 32
```

Train one model at a time if needed:

```powershell
python train.py --model custom --data dataset/split --epochs 15
python train.py --model mobilenetv2 --data dataset/split --epochs 10
```

**MobileNetV2 note:** the first run may download ImageNet pretrained weights. An internet connection is needed for that initial download. To train without pretrained weights, use `--no-pretrained` (usually slower to converge and may perform worse on a small dataset).

For a CPU-only laptop, start with `--batch-size 8` and `--epochs 5`.

## Step 3 – Evaluate saved models

```powershell
python evaluate.py --model all --data dataset/split
```

The evaluation uses the held-out test split and saves:
- accuracy, precision, recall and F1-score
- per-class classification report
- confusion matrix image
- predictions CSV

## Output folders

```text
outputs/
├── custom/
│   ├── best_model.pth
│   ├── history.csv
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── metrics.json
│   └── predictions.csv
└── mobilenetv2/
    └── (same files)
```

`outputs/comparison.csv` is created after both models have been evaluated.

## CNN architecture overview

### Custom CNN
`Input (224×224×3) → Conv(32) + BatchNorm + ReLU → MaxPool → Conv(64) + BatchNorm + ReLU → MaxPool → Conv(128) + BatchNorm + ReLU → MaxPool → Conv(256) + BatchNorm + ReLU → Adaptive Average Pool → Dropout → Fully Connected → Softmax classes`

### MobileNetV2
`Input (224×224×3) → MobileNetV2 inverted-residual/depthwise-separable feature extractor → Global Average Pool → Dropout → Fully Connected class layer → Softmax classes`

The code uses logits with `CrossEntropyLoss`; softmax is applied only when probabilities are needed for reporting.

## Important for your report

- Do not invent or copy sample accuracy values. Run the programs and report the actual values saved in `metrics.json`.
- Training time is measured by the program and saved in the metrics.
- The dataset split is fixed using seed 42 for reproducibility.
- The test set is kept separate from training and validation.
- Results depend on dataset quality, class balance, hardware, and training settings.
- The program reports model parameter counts as a simple model-complexity measure.
- This is a supervised image-classification prototype, not a complete industrial inspection system.

## Useful commands

```powershell
# Check installed PyTorch and GPU
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Train custom CNN
python train.py --model custom --data dataset/split --epochs 15 --batch-size 16

# Train MobileNetV2
python train.py --model mobilenetv2 --data dataset/split --epochs 10 --batch-size 16

# Evaluate both
python evaluate.py --model all --data dataset/split
```
