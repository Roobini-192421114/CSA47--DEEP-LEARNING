import math

P = [0.9, 0.1]
Q = [0.5, 0.5]

def entropy(probabilities):
    result = 0
    for p in probabilities:
        result -= p * math.log2(p)
    return result

h1 = entropy(P)
h2 = entropy(Q)

print("Patient 1 Entropy:", round(h1, 4), "bits")
print("Patient 2 Entropy:", round(h2, 4), "bits")

if h1 > h2:
    print("Patient 1 has greater uncertainty")
else:
    print("Patient 2 has greater uncertainty")
