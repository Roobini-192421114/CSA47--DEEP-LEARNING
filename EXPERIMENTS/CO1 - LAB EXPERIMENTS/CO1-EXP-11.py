import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

x = np.arange(21)
p = binom.pmf(x, 20, 0.1)

print("P(X=3):", binom.pmf(3, 20, 0.1))

plt.bar(x, p)
plt.xlabel("Defective Components")
plt.ylabel("Probability")
plt.title("Binomial Distribution")
plt.show()
