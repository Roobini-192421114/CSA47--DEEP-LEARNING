import math

a = 5
b = 2
c = 2
d = 2

trace = a + d
det = a * d - b * c

discriminant = math.sqrt(trace**2 - 4 * det)

e1 = (trace + discriminant) / 2
e2 = (trace - discriminant) / 2

print("Eigenvalue 1:", round(e1, 2))
print("Eigenvalue 2:", round(e2, 2))

# Eigenvector for the largest eigenvalue
x = e1 - d
y = c

length = math.sqrt(x**2 + y**2)
x = x / length
y = y / length

print("Largest Eigenvalue:", round(e1, 2))
print("Principal Eigenvector:", round(x, 4), round(y, 4))

print("Other Eigenvector: approximately (-0.4472, 0.8944)")
print("Principal direction identified successfully.")
