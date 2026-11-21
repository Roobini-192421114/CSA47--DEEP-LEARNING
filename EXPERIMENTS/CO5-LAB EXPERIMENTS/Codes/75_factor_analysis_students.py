import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
np.random.seed(11); n=300; latent=np.random.randn(n,2); X=np.c_[latent[:,0]+.2*np.random.randn(n),latent[:,0]+.3*np.random.randn(n),latent[:,1]+.3*np.random.randn(n),latent[:,1]+.2*np.random.randn(n),.5*latent[:,0]+.5*latent[:,1]+.3*np.random.randn(n)]
f=FactorAnalysis(n_components=2,random_state=11).fit(X); L=f.components_.T
plt.imshow(L,aspect='auto'); plt.colorbar(label='Loading'); plt.xticks(range(2),['Factor 1','Factor 2']); plt.yticks(range(5),['Marks','Attendance','Assignments','Participation','Exam']); plt.title('Student performance factor loadings'); plt.tight_layout(); plt.savefig('75_output.png'); print(L)
