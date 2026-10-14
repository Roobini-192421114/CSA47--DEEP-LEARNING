
import numpy as np


roll_numbers = np.arange(1, 61)


time_points = np.linspace(0, 1, 50)


zeros_vector = np.zeros(8)


ones_vector = np.ones(8)


print("Roll numbers:", roll_numbers)
print("Time points:", time_points)
print("Zeros vector:", zeros_vector)
print("Ones vector:", ones_vector)


print("\nVector details:")
print("Roll numbers - Length:", len(roll_numbers),
      "Dtype:", roll_numbers.dtype)

print("Time points - Length:", len(time_points),
      "Dtype:", time_points.dtype)

print("Zeros - Length:", len(zeros_vector),
      "Dtype:", zeros_vector.dtype)

print("Ones - Length:", len(ones_vector),
      "Dtype:", ones_vector.dtype)
