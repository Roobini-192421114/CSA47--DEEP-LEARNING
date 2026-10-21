import math

P = [0.5, 0.3, 0.2]
Q = [0.4, 0.4, 0.2]

kl = 0

for i in range(3):
    kl += P[i] * math.log2(P[i] / Q[i])

print("Distribution P:", P)
print("Distribution Q:", Q)
print("KL Divergence:", round(kl, 4), "bits")
