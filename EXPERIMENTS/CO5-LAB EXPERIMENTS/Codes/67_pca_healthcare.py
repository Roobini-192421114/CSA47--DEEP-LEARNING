import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
np.random.seed(10); X=np.random.randn(300,8); X[:,1]=.7*X[:,0]+.3*X[:,1]; X[:,2]=.5*X[:,0]+.5*X[:,2]
X=StandardScaler().fit_transform(X); p=PCA().fit_transform(X); ev=PCA().fit(X).explained_variance_ratio_
plt.bar(range(1,9),ev); plt.plot(range(1,9),np.cumsum(ev),marker='o'); plt.xlabel('Principal component'); plt.ylabel('Explained variance'); plt.title('Synthetic healthcare attributes PCA'); plt.tight_layout(); plt.savefig('67_output.png'); print('Variance:',ev)
