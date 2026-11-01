# Q7 - Redesign and Validation

## Problem identified
Q5 showed a significant test accuracy decrease when images were
shifted by two pixels. The CNN was not fully robust to translation.

## Proposed redesign
Randomly translate training images by up to two pixels, with empty
pixels filled with zero. The architecture, training subset, optimizer,
epoch count and evaluation set remain the same for both experiments.

## Measured results
                    model  test_loss  clean_test_accuracy  training_time_seconds  trainable_parameters
             Baseline CNN   0.406485                0.852               7.761908                 67530
Translation-Augmented CNN   0.488166                0.813               8.697332                 67530

## Translation results
                    model  shift_x  shift_y  accuracy
             Baseline CNN        0        0     0.852
             Baseline CNN        2        0     0.606
             Baseline CNN       -2        0     0.625
             Baseline CNN        0        2     0.701
             Baseline CNN        0       -2     0.767
Translation-Augmented CNN        0        0     0.813
Translation-Augmented CNN        2        0     0.796
Translation-Augmented CNN       -2        0     0.782
Translation-Augmented CNN        0        2     0.783
Translation-Augmented CNN        0       -2     0.776

## Interpretation
Compare clean accuracy and shifted-image accuracy. The redesign is
useful only if the measured results show an improvement in robustness
without an unacceptable loss in clean-image accuracy.

This is a single-seed experiment on a limited subset. Additional runs
and a larger evaluation are needed for stronger conclusions.
