import math

P = [0.6, 0.3, 0.1]
weather = ["Sunny", "Rainy", "Cloudy"]

entropy = 0

for p in P:
    entropy -= p * math.log2(p)

print("Weather Probabilities:")
for i in range(3):
    print(weather[i], ":", P[i])

print("Entropy:", round(entropy, 4), "bits")
