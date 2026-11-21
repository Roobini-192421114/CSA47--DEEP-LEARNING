import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X,y=make_blobs(n_samples=400,n_features=8,centers=4,random_state=5); Z=PCA(2).fit_transform(StandardScaler().fit_transform(X))
plt.scatter(Z[:,0],Z[:,1],c=y); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('High-dimensional patterns in 2D PCA'); plt.tight_layout(); plt.savefig('69_output.png')
