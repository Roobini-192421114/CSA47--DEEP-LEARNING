import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.preprocessing import StandardScaler
np.random.seed(13); X=np.random.randn(300,10); X[:,1:4]+=X[:,:1]*2; X=StandardScaler().fit_transform(X)
p=PCA(3).fit(X); f=FactorAnalysis(3,random_state=13).fit(X); recp=p.inverse_transform(p.transform(X)); recf=f.transform(X) @ f.components_ + f.mean_; errs=[np.mean((X-recp)**2),np.mean((X-recf)**2)]
plt.bar(['PCA','Factor Analysis'],errs); plt.ylabel('Mean squared reconstruction error'); plt.title('Reconstruction error comparison'); plt.tight_layout(); plt.savefig('77_output.png'); print(errs)
