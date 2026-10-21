import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

x = np.arange(13)
p = poisson.pmf(x, 4)

print("P(X=6):", poisson.pmf(6, 4))

plt.bar(x, p)
plt.xlabel("Customer Arrivals")
plt.ylabel("Probability")
plt.title("Poisson Distribution")
plt.show()
