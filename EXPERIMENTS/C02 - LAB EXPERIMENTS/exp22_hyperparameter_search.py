import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

pipeline = make_pipeline(StandardScaler(), SVC())
grid_params = {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]}
grid = GridSearchCV(pipeline, grid_params, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)
print("Grid search best parameters:", grid.best_params_)
print("Grid search test accuracy:", round(accuracy_score(y_test, grid.predict(X_test)), 4))

random_params = {"svc__C": np.logspace(-2, 2, 15),
                 "svc__gamma": ["scale", 0.001, 0.01, 0.1, 1]}
random_search = RandomizedSearchCV(
    pipeline, random_params, n_iter=10, cv=3, scoring="accuracy", random_state=42
)
random_search.fit(X_train, y_train)
print("\nRandom search best parameters:", random_search.best_params_)
print("Random search test accuracy:", round(accuracy_score(y_test, random_search.predict(X_test)), 4))

# Small Bayesian-optimization-style demonstration:
# Evaluate a set of C values, then use a Gaussian process to estimate the best region.
c_values = np.array([0.03, 0.1, 0.3, 1, 3, 10, 30]).reshape(-1, 1)
scores = []
for c in c_values[:, 0]:
    candidate = make_pipeline(StandardScaler(), SVC(C=float(c), gamma="scale"))
    candidate.fit(X_train, y_train)
    scores.append(accuracy_score(y_test, candidate.predict(X_test)))
scores = np.array(scores)

gp = GaussianProcessRegressor(kernel=Matern() + WhiteKernel(), normalize_y=True,
                              random_state=42)
gp.fit(np.log10(c_values), scores)
candidates = np.linspace(-2, 2, 100).reshape(-1, 1)
mean, std = gp.predict(candidates, return_std=True)
next_log_c = candidates[np.argmax(mean + 0.5 * std), 0]
print("\nBayesian-style GP suggested C:", round(10 ** next_log_c, 4))
print("Best observed C:", c_values[np.argmax(scores), 0])
print("Note: this is an educational GP-based search illustration, not a full production Bayesian optimizer.")
