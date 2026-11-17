import numpy as np

# 3D robot position and movement vectors
position = np.array([2, 3, 1])
movement = np.array([4, -1, 2])
scale = 2

print("Original position:", position)
print("Movement vector:", movement)
print("Vector addition (new position):", position + movement)
print("Vector subtraction:", position - movement)
print(f"Scalar multiplication ({scale} x movement):", scale * movement)
