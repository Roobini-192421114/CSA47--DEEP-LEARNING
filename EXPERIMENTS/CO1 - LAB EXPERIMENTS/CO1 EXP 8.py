a = 4
b = 2
c = 2
d = 4

trace = a + d
det = a * d - b * c

discriminant = trace**2 - 4 * det

eigenvalue1 = (trace + discriminant**0.5) / 2
eigenvalue2 = (trace - discriminant**0.5) / 2

print("Matrix A:")
print("[", a, b, "]")
print("[", c, d, "]")

print("Eigenvalue 1:", eigenvalue1)
print("Eigenvalue 2:", eigenvalue2)
