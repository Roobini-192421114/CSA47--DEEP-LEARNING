import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
samples = np.random.normal(loc=25, scale=4, size=1000)

print("First 10 observations:", np.round(samples[:10], 2))
print("Sample mean:", round(np.mean(samples), 2))
print("Sample standard deviation:", round(np.std(samples), 2))
print("Estimated probability temperature is between 21 and 29:",
      round(np.mean((samples >= 21) & (samples <= 29)), 4))

plt.hist(samples, bins=25, density=True, edgecolor="black")
plt.title("Generated Temperature Observations")
plt.xlabel("Temperature")
plt.ylabel("Estimated density")
plt.tight_layout()
plt.show()
