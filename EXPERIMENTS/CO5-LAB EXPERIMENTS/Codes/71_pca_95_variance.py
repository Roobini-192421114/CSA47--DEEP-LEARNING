import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X,_=make_classification(n_samples=500,n_features=25,n_informative=8,n_redundant=8,random_state=1); X=StandardScaler().fit_transform(X); p=PCA().fit(X); cum=np.cumsum(p.explained_variance_ratio_); n=np.searchsorted(cum,.95)+1
plt.plot(range(1,len(cum)+1),cum,marker='o'); plt.axhline(.95,linestyle='--'); plt.axvline(n,linestyle='--'); plt.xlabel('Components'); plt.ylabel('Cumulative explained variance'); plt.title('PCA component selection (95%)'); plt.tight_layout(); plt.savefig('71_output.png'); print('Components required:',n)
