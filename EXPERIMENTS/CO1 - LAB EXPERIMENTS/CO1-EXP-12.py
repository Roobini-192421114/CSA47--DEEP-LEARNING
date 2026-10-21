import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean, sd = 70, 10
prob = norm.cdf(80, mean, sd) - norm.cdf(60, mean, sd)
print("Probability:", prob)

x = np.linspace(40, 100, 200)
y = norm.pdf(x, mean, sd)

plt.plot(x, y)
plt.fill_between(x, y, where=(x >= 60) & (x <= 80), alpha=0.4)
plt.title("Normal Distribution")
plt.show()
