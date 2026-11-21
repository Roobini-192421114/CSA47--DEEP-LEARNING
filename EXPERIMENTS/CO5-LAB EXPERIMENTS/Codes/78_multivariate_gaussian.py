import numpy as np, matplotlib.pyplot as plt
np.random.seed(14); mean=np.array([2,-1]); cov=np.array([[2,.8],[.8,1.2]]); X=np.random.multivariate_normal(mean,cov,1000); est_mean=X.mean(0); est_cov=np.cov(X.T)
plt.scatter(X[:,0],X[:,1],s=10,alpha=.5); plt.xlabel('Feature 1'); plt.ylabel('Feature 2'); plt.title('Synthetic multivariate Gaussian'); plt.tight_layout(); plt.savefig('78_output.png'); print('Estimated mean:',est_mean,'\nCovariance:',est_cov)
