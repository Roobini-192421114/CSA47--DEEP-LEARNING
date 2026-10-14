
import numpy as np


usd = np.array([10, 20, 30, 40, 50])


rate = 83.5


inr = usd * rate


discounted = inr * 0.9

print("USD Prices:", usd)
print("INR Prices:", inr)
print("Discounted Prices:", discounted)


expected = 0.9 * 83.5 * usd

print("Expected Prices:", expected)
print("Check:", np.allclose(discounted, expected))
