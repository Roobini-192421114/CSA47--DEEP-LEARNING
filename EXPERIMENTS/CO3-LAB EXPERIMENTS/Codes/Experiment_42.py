# Experiment 42: CNN filter size and number of filters illustration
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import convolve2d
img=np.zeros((20,20)); img[5:15,5:15]=1; img[8:12,8:12]=0
kernels=[np.array([[1,0,-1],[1,0,-1],[1,0,-1]]),np.ones((5,5))/25]
maps=[convolve2d(img,k,mode='same') for k in kernels]
fig,ax=plt.subplots(1,3,figsize=(9,3)); ax[0].imshow(img,cmap='gray'); ax[0].set_title('Input')
for i in range(2):ax[i+1].imshow(maps[i],cmap='coolwarm'); ax[i+1].set_title(f'Filter {i+1}: {kernels[i].shape[0]}x{kernels[i].shape[0]}')
for a in ax:a.axis('off')
plt.tight_layout(); plt.savefig('../Outputs/42_filter_size_effect.png'); print('Filter effects plotted')
