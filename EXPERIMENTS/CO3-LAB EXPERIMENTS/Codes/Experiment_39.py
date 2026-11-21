# Experiment 39: Convolution and pooling feature map demonstration
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import convolve2d
img=np.zeros((12,12)); img[3:9,3:9]=1; img[5:7,5:7]=0
kernel=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
conv=convolve2d(img,kernel,mode='same'); pool=np.array([[np.max(conv[i:i+2,j:j+2]) for j in range(0,12,2)] for i in range(0,12,2)])
fig,ax=plt.subplots(1,3,figsize=(9,3)); ax[0].imshow(img,cmap='gray'); ax[0].set_title('Input'); ax[1].imshow(conv,cmap='coolwarm'); ax[1].set_title('Convolution'); ax[2].imshow(pool,cmap='coolwarm'); ax[2].set_title('Max pooling')
for a in ax:a.axis('off')
plt.tight_layout(); plt.savefig('../Outputs/39_conv_pool_maps.png'); print('Feature maps created')
