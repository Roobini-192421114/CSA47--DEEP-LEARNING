import numpy as np

# Column-stochastic transition matrix: columns are source pages
P = np.array([
    [0.0, 0.5, 0.5],
    [1.0, 0.0, 0.5],
    [0.0, 0.5, 0.0]
])

print("Transition matrix:\n", P)
values, vectors = np.linalg.eig(P)
print("\nEigenvalues:", np.round(values, 4))

# Eigenvector associated with eigenvalue closest to 1
index = np.argmin(np.abs(values - 1))
ranking = np.real(vectors[:, index])
ranking = np.abs(ranking)
ranking = ranking / ranking.sum()

print("Principal eigenvector (normalized page scores):", np.round(ranking, 4))
for i, score in enumerate(ranking, start=1):
    print(f"Page {i}: {score:.4f}")
print("Highest score indicates the most important page in this small network.")
