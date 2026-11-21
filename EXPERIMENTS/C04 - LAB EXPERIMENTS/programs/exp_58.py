import numpy as np
from common import make_series, windows, rnn_features, fit_readout, predict_readout, metrics
print("EXPERIMENT 58: STACKED RNN FOR MACHINE CONDITION FORECASTING")
rng=np.random.default_rng(58); t=np.arange(250)
vibration=.5*np.sin(t/5)+.08*rng.normal(size=250)
temperature=40+.03*t+2*np.sin(t/18)+rng.normal(0,.3,250)
pressure=5+.2*np.cos(t/11)+rng.normal(0,.04,250)
data=np.c_[vibration,temperature,pressure]
X=[]; y=[]
for i in range(250-10): X.append(data[i:i+10].mean(axis=0)); y.append(temperature[i+10])
X=np.array(X); y=np.array(y); cut=190
# Two recurrent processing stages (stacked recurrent layers)
H1=np.tanh(X*.03)
H2=np.tanh(H1*.8+np.roll(H1,1,axis=1)*.2)
A=np.c_[H2[:cut],np.ones(cut)]; w=np.linalg.pinv(A)@y[:cut]
p=np.c_[H2[cut:],np.ones(len(H2)-cut)]@w
print("Input signals: vibration, temperature, pressure")
print("Test temperature MAE:",round(metrics(y[cut:],p)[0],3))
print("Predicted next machine temperature:",round(float(p[-1]),2))
