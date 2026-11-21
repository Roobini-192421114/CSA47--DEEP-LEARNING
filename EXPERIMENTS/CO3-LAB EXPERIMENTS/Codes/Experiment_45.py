# Experiment 45: Image augmentation impact demonstration
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(45); base=np.zeros((16,16)); base[4:12,5:11]=1
aug=[base,np.rot90(base),np.roll(base,2,axis=0),base+ rng.normal(0,.12,base.shape)]
fig,ax=plt.subplots(1,4,figsize=(8,2))
for i,a in enumerate(ax): a.imshow(aug[i],cmap='gray'); a.set_title(['Original','Rotation','Shift','Noise'][i]); a.axis('off')
plt.tight_layout(); plt.savefig('../Outputs/45_augmentation_examples.png'); print('Augmentation samples generated')
