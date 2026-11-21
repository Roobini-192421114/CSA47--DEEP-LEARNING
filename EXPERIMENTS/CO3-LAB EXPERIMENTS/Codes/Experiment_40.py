# Experiment 40: Max pooling vs average pooling feature effects
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import convolve2d
rng=np.random.default_rng(40); img=np.zeros((16,16)); img[4:12,5:11]=1; img+=rng.normal(0,.18,img.shape)
kernel=np.ones((3,3))/9; feat=convolve2d(img,kernel,mode='same')
maxp=np.array([[np.max(feat[i:i+2,j:j+2]) for j in range(0,16,2)] for i in range(0,16,2)])
avgp=np.array([[np.mean(feat[i:i+2,j:j+2]) for j in range(0,16,2)] for i in range(0,16,2)])
fig,ax=plt.subplots(1,3,figsize=(9,3)); ax[0].imshow(img,cmap='gray'); ax[0].set_title('Input'); ax[1].imshow(maxp,cmap='gray'); ax[1].set_title('Max pooling'); ax[2].imshow(avgp,cmap='gray'); ax[2].set_title('Average pooling')
for a in ax:a.axis('off')
plt.tight_layout(); plt.savefig('../Outputs/40_pooling_comparison.png'); print('Pooling comparison plot saved')
