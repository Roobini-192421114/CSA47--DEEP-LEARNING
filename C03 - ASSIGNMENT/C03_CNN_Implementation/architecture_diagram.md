# CNN Architecture Diagrams

## Model 1 – Custom CNN

```text
RGB Image (224 x 224 x 3)
          |
          v
Conv2D 3x3, 32 filters
          |
Batch Normalization + ReLU
          |
Max Pooling 2x2
          |
Conv2D 3x3, 64 filters
          |
Batch Normalization + ReLU
          |
Max Pooling 2x2
          |
Conv2D 3x3, 128 filters
          |
Batch Normalization + ReLU
          |
Max Pooling 2x2
          |
Conv2D 3x3, 256 filters
          |
Batch Normalization + ReLU
          |
Adaptive Average Pooling (1x1)
          |
Flatten + Dropout (0.35)
          |
Fully Connected (256 -> number of classes)
          |
Class scores (logits) -> Softmax for probabilities
```

## Model 2 – MobileNetV2

```text
RGB Image (224 x 224 x 3)
          |
          v
MobileNetV2 feature extractor
(Inverted residual blocks, depthwise separable convolutions,
linear bottlenecks and residual connections)
          |
Global Average Pooling
          |
Dropout (0.30)
          |
Fully Connected classification layer
          |
Class scores (logits) -> Softmax for probabilities
```

## Main design rationale

- **Convolution:** learns local visual patterns such as edges, scratches, spots and texture changes.
- **Batch normalization:** stabilizes feature distributions and can make optimization more reliable.
- **ReLU:** adds non-linearity and helps learn complex visual patterns.
- **Max pooling:** reduces spatial dimensions and computation while retaining strong features.
- **Adaptive/global average pooling:** converts feature maps into a compact feature vector.
- **Dropout:** reduces co-adaptation and helps control overfitting.
- **Fully connected output:** maps learned features to one score per defect class.
- **Cross-entropy loss:** trains the model for mutually exclusive defect categories.
