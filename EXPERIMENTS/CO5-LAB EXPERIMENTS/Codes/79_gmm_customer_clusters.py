import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
X,y=make_blobs(n_samples=450,centers=3,n_features=2,cluster_std=1.1,random_state=15); m=GaussianMixture(3,random_state=15).fit(X); pred=m.predict(X); probs=m.predict_proba(X)
plt.scatter(X[:,0],X[:,1],c=pred); plt.xlabel('Purchase feature 1'); plt.ylabel('Purchase feature 2'); plt.title('GMM customer segments'); plt.tight_layout(); plt.savefig('79_output.png'); print('Sample membership probabilities:',probs[:5])
