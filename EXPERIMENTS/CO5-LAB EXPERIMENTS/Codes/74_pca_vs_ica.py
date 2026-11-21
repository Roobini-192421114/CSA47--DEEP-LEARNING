import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
np.random.seed(9); S=np.c_[np.sin(np.linspace(0,10,600)),np.sign(np.sin(np.linspace(0,18,600)))]; X=S@np.array([[1,2],[.5,1.]]).T; X+=.1*np.random.randn(*X.shape)
p=PCA(2).fit_transform(X); i=FastICA(2,random_state=9).fit_transform(X)
fig,ax=plt.subplots(1,2,figsize=(9,4)); ax[0].scatter(p[:,0],p[:,1]); ax[0].set_title('PCA: uncorrelated components'); ax[1].scatter(i[:,0],i[:,1]); ax[1].set_title('ICA: independent components'); plt.tight_layout(); plt.savefig('74_output.png')
