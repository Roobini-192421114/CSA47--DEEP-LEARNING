import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
np.random.seed(12); z=np.random.randn(400,2); X=np.c_[z[:,0]+.2*np.random.randn(400),z[:,0]+.3*np.random.randn(400),z[:,1]+.2*np.random.randn(400),z[:,1]+.3*np.random.randn(400),z[:,0]+z[:,1]+.3*np.random.randn(400)]
f=FactorAnalysis(2,random_state=12).fit(X); L=f.components_.T
plt.imshow(L,aspect='auto'); plt.colorbar(label='Loading'); plt.xticks(range(2),['Factor 1','Factor 2']); plt.yticks(range(5),['Purchase frequency','Basket size','Discount use','Online activity','Brand affinity']); plt.title('Customer latent-factor loadings'); plt.tight_layout(); plt.savefig('76_output.png'); print(L)
