import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("A x B:\n", A @ B)
print("Transpose of A:\n", A.T)
print("Transpose of B:\n", B.T)
print("\nAnalysis: multiplication combines rows of A with columns of B;")
print("transpose swaps rows and columns.")
