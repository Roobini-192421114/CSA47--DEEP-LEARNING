import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
np.random.seed(3); t=np.linspace(0,8,1000); s1=np.sin(2*np.pi*t); s2=np.sign(np.sin(3*np.pi*t)); S=np.c_[s1,s2]; A=np.array([[1,.5],[.4,1.3]]); M=S@A.T; R=FastICA(n_components=2,random_state=3).fit_transform(M)
fig,ax=plt.subplots(3,1,figsize=(9,6)); ax[0].plot(t,S); ax[0].set_title('Original sources'); ax[1].plot(t,M); ax[1].set_title('Mixed signals'); ax[2].plot(t,R); ax[2].set_title('ICA recovered components'); plt.tight_layout(); plt.savefig('72_output.png')
