import numpy as np

# Example Hill-cipher-style key matrix
key = np.array([[3, 3], [2, 5]], dtype=float)

det = np.linalg.det(key)
rank = np.linalg.matrix_rank(key)
print("Encryption matrix:\n", key)
print("Determinant:", round(det, 4))
print("Rank:", rank)

if not np.isclose(det, 0):
    inverse = np.linalg.inv(key)
    print("Inverse matrix:\n", np.round(inverse, 4))
    print("Suitable over real numbers: YES (matrix is invertible)")
else:
    print("Inverse does not exist.")
    print("Suitable for reversible matrix encryption: NO")

print("Note: practical Hill cipher encryption requires an inverse modulo the chosen alphabet size.")
