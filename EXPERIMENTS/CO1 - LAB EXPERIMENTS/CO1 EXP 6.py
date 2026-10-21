import numpy as np

hours = [2, 4, 6, 8, 10]
marks = [35, 45, 60, 75, 85]

covariance = np.cov(hours, marks)[0, 1]

print("Study Hours:", hours)
print("Examination Marks:", marks)
print("Sample Covariance:", covariance)
