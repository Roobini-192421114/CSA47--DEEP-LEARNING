# Experiment 44: Batch-normalization training stability illustration
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(44); epochs=np.arange(1,41)
loss_no=1.8*np.exp(-epochs/12)+.15*np.abs(rng.normal(size=40))
loss_bn=1.7*np.exp(-epochs/10)+.04*np.abs(rng.normal(size=40))
plt.figure(figsize=(7,4)); plt.plot(epochs,loss_no,label='Without batch norm'); plt.plot(epochs,loss_bn,label='With batch norm'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Batch normalization stability (illustration)'); plt.legend(); plt.tight_layout(); plt.savefig('../Outputs/44_batch_normalization.png'); print('Training loss curves saved')
