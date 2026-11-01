# CSA47 Deep Learning CO2 Assignment — Q1

## Scope
Q1 compares Sigmoid, Tanh and ReLU hidden units as network depth increases. The script tests:
1. `shallow`: 2 hidden layers
2. `deep`: 8 hidden layers
3. `poor_init`: 8 hidden layers with deliberately large weights to expose unstable/saturated behaviour

Dataset: Fashion-MNIST (70,000 28×28 grayscale images; 60,000 official training images and 10,000 official test images). This experiment uses a reproducible subset of the official training partition: 6,000 training examples, 1,000 validation examples, and 1,000 test examples. The remaining official training examples are not used in this Q1 run. The test set is held out and is not used for model selection.

Note: Fashion-MNIST is an image dataset, whereas the question's motivating scenario discusses tabular credit-card transactions. The experiment therefore tests the activation/depth principles on a public benchmark; state this limitation clearly in your report. If your instructor requires a tabular fraud dataset, confirm before replacing the dataset.

## Windows / VS Code setup
Open PowerShell in this folder.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run Q1
Start with a quick smoke test:
```powershell
python src/q1_activation_depth.py --epochs 2 --runs 1 --train-limit 1000 --val-limit 300 --test-limit 300
```

Then run the fuller experiment:
```powershell
python src/q1_activation_depth.py --epochs 5 --runs 3 --train-limit 6000 --val-limit 1000 --test-limit 1000
```

First run downloads Fashion-MNIST. Internet is needed for the initial download.

## Outputs
The script writes actual measurements into `outputs/`:
- `q1_run_results.csv`: one row per run/configuration/activation
- `q1_summary.csv`: mean and standard deviation across runs
- `q1_epoch_history.csv`: training/validation metrics per epoch
- `q1_gradient_norms.png`: layer-wise gradient norms
- `q1_training_curves.png`: training loss curves
- `q1_dead_relu.png`: dead-ReLU fraction where applicable
- `q1_metadata.json`: dataset sizes, settings and random seeds

Do not write expected values as if they were measured. Use the generated CSV/plots in the report. The script logs errors rather than silently treating failed runs as results.

## Q1 analysis to write after running
Compare:
- Early-layer gradient norms as depth increases for sigmoid, tanh and ReLU.
- Dead-ReLU fraction for ReLU networks.
- Training loss and validation accuracy/precision/recall.
- Predicted versus observed behaviour.
- Recommended activation and initialization for at least 8 hidden layers, supported by your measured results.

Keep all output files for the final assignment evidence.
