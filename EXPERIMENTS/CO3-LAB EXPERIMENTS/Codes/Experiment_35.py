# Experiment 35: Dropout-like regularization and overfitting illustration
import numpy as np, matplotlib.pyplot as plt
epochs=np.arange(1,51); train=1-np.exp(-epochs/8); val=0.78*(1-np.exp(-epochs/8))+0.08*np.sin(epochs/6)
val=np.clip(val,0,1)
val_drop=0.76*(1-np.exp(-epochs/10))+0.02*np.sin(epochs/7)
plt.figure(figsize=(7,4)); plt.plot(epochs,train,label='Training (no dropout)'); plt.plot(epochs,val,label='Validation (no dropout)'); plt.plot(epochs,val_drop,label='Validation (dropout-like)'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Dropout and overfitting'); plt.legend(); plt.tight_layout(); plt.savefig('../Outputs/35_dropout_effect.png'); print('Plot generated: dropout effect')
