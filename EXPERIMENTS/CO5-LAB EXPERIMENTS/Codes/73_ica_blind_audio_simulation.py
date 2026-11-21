import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
np.random.seed(8); t=np.linspace(0,5,1200); voice1=np.sin(2*np.pi*(180+20*np.sin(t))*t); voice2=np.sin(2*np.pi*260*t)+.4*np.sin(2*np.pi*520*t); S=np.c_[voice1,voice2]; mix=S@np.array([[1,.6],[.3,1.1]]).T; sep=FastICA(n_components=2,random_state=8).fit_transform(mix)
fig,ax=plt.subplots(3,1,figsize=(9,6)); ax[0].plot(t,S); ax[0].set_title('Simulated source signals'); ax[1].plot(t,mix); ax[1].set_title('Microphone mixtures'); ax[2].plot(t,sep); ax[2].set_title('Separated ICA components'); plt.tight_layout(); plt.savefig('73_output.png'); print('Note: synthetic tones, not real voice recordings.')
