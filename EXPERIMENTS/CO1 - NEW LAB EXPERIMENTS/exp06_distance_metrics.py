import numpy as np

customer_1 = np.array([2, 5, 3])
customer_2 = np.array([6, 2, 3])

euclidean = np.linalg.norm(customer_1 - customer_2)
manhattan = np.sum(np.abs(customer_1 - customer_2))

print("Customer 1:", customer_1)
print("Customer 2:", customer_2)
print("Euclidean distance:", round(euclidean, 4))
print("Manhattan distance:", manhattan)
print("\nEuclidean measures straight-line distance.")
print("Manhattan measures total absolute difference across features.")
