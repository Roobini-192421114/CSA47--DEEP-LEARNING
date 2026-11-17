import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 300)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

plt.plot(x, sigmoid, label="Sigmoid")
plt.plot(x, tanh, label="Tanh")
plt.plot(x, relu, label="ReLU")
plt.title("Activation Function Outputs")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Sigmoid range: 0 to 1")
print("Tanh range: -1 to 1")
print("ReLU returns 0 for negative inputs and x for positive inputs.")
