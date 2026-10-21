import matplotlib.pyplot as plt

prior = 0.02
sensitivity = 0.95
false_positive = 0.05

posterior = (sensitivity * prior) / (
    sensitivity * prior +
    false_positive * (1 - prior)
)

print("Posterior probability:", posterior)

plt.bar(["Prior", "After Positive Test"],
        [prior, posterior])
plt.ylabel("Probability")
plt.show()
