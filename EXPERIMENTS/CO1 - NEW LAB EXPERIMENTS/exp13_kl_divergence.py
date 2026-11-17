import numpy as np

# Actual/reference distribution P and predicted distribution Q
P = np.array([0.50, 0.30, 0.20])
Q = np.array([0.40, 0.40, 0.20])

kl_divergence = np.sum(P * np.log(P / Q))  # natural logarithm: nats

print("Reference distribution P:", P)
print("Predicted distribution Q:", Q)
print("KL divergence D(P || Q):", round(kl_divergence, 6), "nats")
print("KL divergence is non-negative and is zero when the distributions match.")
print("It measures the information difference when Q is used to approximate P.")
