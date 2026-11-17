import numpy as np

# Actual labels (1 = spam, 0 = not spam)
actual = np.array([1, 0, 1, 1, 0])
# Model's predicted probability that each email is spam
predicted = np.array([0.90, 0.20, 0.70, 0.60, 0.10])

epsilon = 1e-15
predicted = np.clip(predicted, epsilon, 1 - epsilon)
losses = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
mean_loss = np.mean(losses)

print("Actual labels:", actual)
print("Predicted probabilities:", predicted)
print("Individual cross-entropy losses:", np.round(losses, 4))
print("Mean binary cross-entropy loss:", round(mean_loss, 4))
print("Lower loss indicates predictions closer to the actual labels.")
