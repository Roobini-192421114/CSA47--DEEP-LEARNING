# Q6 - Architecture Design

Dataset: Fashion-MNIST (28x28 grayscale, 10 classes).
Training subset: 4000; validation subset: 1000;
test subset: 1000.
Epochs: 4; batch size: 64; seed: 42.

## Architecture comparison
architecture  trainable_parameters  estimated_MACs_per_image  receptive_field_pixels  weight_file_size_MB  training_time_seconds  inference_time_ms_per_image  test_loss  test_accuracy  best_validation_accuracy
    SmallCNN                 15770                     72128                       4             0.062613               4.059719                     0.153476   0.576875          0.794                     0.793
   MediumCNN                 20490                   1031744                      10             0.081311               5.438067                     0.265354   0.519775          0.811                     0.803
     DeepCNN                 31386                   3731328                      24             0.124028               7.553860                     0.595888   0.587778          0.776                     0.776

## Receptive field calculation
For a layer with kernel size k and previous effective stride j:
RF_new = RF_previous + (k - 1) * j.
The effective stride is multiplied by the layer stride.
These values describe the feature extractor's theoretical receptive field.

## MACs
MACs estimate multiply-accumulate operations for a single image.
They exclude bias additions, activations and pooling comparisons.

## Deployment boundary
This experiment uses 28x28 images and CPU inference. A deployment decision
should consider validation accuracy, latency, model file size, available
memory and the target hardware. The fastest or smallest model is not
automatically the best model.

## Outputs
- q6_architecture_comparison.csv
- q6_training_history.csv
- q6_validation_accuracy.png
- q6_training_loss.png
- q6_metadata.json
- model weight files for each architecture

Results are from one seed and four epochs per model. For stronger evidence,
repeat with at least three seeds and report mean and standard deviation.
