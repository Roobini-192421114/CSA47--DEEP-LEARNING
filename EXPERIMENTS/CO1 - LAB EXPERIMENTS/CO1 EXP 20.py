import math

Y = [0, 1, 0, 0]
P = [0.1, 0.6, 0.2, 0.1]

cross_entropy = 0
kl = 0

for i in range(4):
    if Y[i] == 1:
        cross_entropy -= math.log(P[i])
        kl += Y[i] * math.log2(Y[i] / P[i])

print("Actual Distribution:", Y)
print("Predicted Distribution:", P)
print("Cross-Entropy Loss:", round(cross_entropy, 4))
print("KL Divergence:", round(kl, 4), "bits")

print("Actual vs Predicted:")
for i in range(4):
    print("Class", i + 1, ":", Y[i], "vs", P[i])
