import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X,y=load_iris(return_X_y=True); Z=PCA(2).fit_transform(StandardScaler().fit_transform(X))
plt.scatter(Z[:,0],Z[:,1],c=y); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Iris dataset after PCA'); plt.tight_layout(); plt.savefig('68_output.png'); print('2D variance captured:',PCA(2).fit(StandardScaler().fit_transform(X)).explained_variance_ratio_.sum())
