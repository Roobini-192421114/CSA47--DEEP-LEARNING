import math
from collections import Counter

# Example loan applicant labels: Low-risk and High-risk
labels = ["Low", "Low", "High", "Low", "High", "Low", "Low", "High", "Low", "High"]

counts = Counter(labels)
total = len(labels)
entropy = 0.0

for label, count in counts.items():
    probability = count / total
    entropy -= probability * math.log2(probability)
    print(f"{label}: count={count}, probability={probability:.2f}")

print("Total applicants:", total)
print("Dataset entropy:", round(entropy, 4), "bits")
print("Maximum entropy for two balanced classes: 1 bit")
print("Higher entropy means greater uncertainty in the class labels.")
