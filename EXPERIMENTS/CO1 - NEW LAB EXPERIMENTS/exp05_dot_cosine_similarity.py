import numpy as np

customer_a = np.array([3, 1, 2, 0])
customer_b = np.array([2, 1, 2, 1])
customer_c = np.array([0, 4, 1, 3])

def compare(x, y):
    dot = np.dot(x, y)
    cosine = dot / (np.linalg.norm(x) * np.linalg.norm(y))
    return dot, cosine

for name, vector in [("Customer B", customer_b), ("Customer C", customer_c)]:
    dot, similarity = compare(customer_a, vector)
    print(f"Customer A vs {name}")
    print("Dot product:", dot)
    print("Cosine similarity:", round(similarity, 4))
    print()

print("A cosine similarity closer to 1 means more similar preferences.")
