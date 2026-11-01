# Q5 - CNN Design

## Dataset and setup
- Dataset: Fashion-MNIST, 28x28 grayscale images, 10 classes.
- Training subset: 5000 images.
- Validation subset: 1000 images.
- Test subset: 1000 images.
- Epochs: 5; batch size: 64; random seed: 42.
- Device: cpu.

## Convolution output calculation
Output size = floor((N + 2P - K) / S) + 1.

For N=28, P=1, K=3, S=1:
(28 + 2 - 3) / 1 + 1 = 28.
Therefore the first convolution produces 28x28 feature maps.

2x2 max pooling with stride 2 produces 14x14 maps.

## First-layer parameter counts
Fully connected layer: 784*128 + 128 = 100480.
CNN convolution: 16*(1*3*3) + 16 = 160.

## Measured comparison
          model  total_parameters  first_layer_parameters  test_accuracy  training_seconds
Fully Connected            101770                  100480          0.824          4.774544
            CNN             67530                     160          0.846          7.982124

## Translation robustness
See q5_translation_results.csv for the accuracy of both models under
horizontal and vertical shifts of two pixels.

## Figures
- q5_training_curves.png
- q5_learned_filters.png
- q5_feature_maps.png

## Interpretation
Convolution uses the same filter weights at every spatial position,
which greatly reduces first-layer parameters and encourages detection
of local patterns. Pooling reduces spatial resolution and provides some
tolerance to small shifts, but does not guarantee complete translation
invariance. The actual performance conclusions must be based on the
measured CSV outputs.
