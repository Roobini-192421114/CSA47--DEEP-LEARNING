import numpy as np

marks = np.array([78, 85, 90, 67, 88, 92, 75, 80, 85, 90])

mean = np.mean(marks)
variance = np.var(marks)          # population variance
std_dev = np.std(marks)           # population standard deviation

print("Student marks:", marks)
print("Mean:", round(mean, 2))
print("Variance:", round(variance, 2))
print("Standard deviation:", round(std_dev, 2))
print("\nInterpretation: a smaller standard deviation means marks are more consistent.")
