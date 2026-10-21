import math

a = 3
b = 1
c = 1
d = 3

trace = a + d
det = a * d - b * c

discriminant = math.sqrt(trace**2 - 4 * det)

e1 = (trace + discriminant) / 2
e2 = (trace - discriminant) / 2

print("Eigenvalues:", e2, e1)

print("Eigenvectors:")
print("For eigenvalue", e2, ": [1, -1]")
print("For eigenvalue", e1, ": [1, 1]")

print("Eigenvectors are plotted as coordinate directions:")
print("Vector 1: (1, -1)")
print("Vector 2: (1, 1)")
