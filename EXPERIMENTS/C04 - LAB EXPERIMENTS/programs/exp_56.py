import numpy as np
from common import metrics
print("EXPERIMENT 56: GRU WEATHER FORECASTING")
rng=np.random.default_rng(56); n=220; t=np.arange(n)
temp=25+5*np.sin(t/12)+rng.normal(0,.5,n)
humidity=60+10*np.cos(t/15)+rng.normal(0,1,n)
pressure=1010+3*np.sin(t/20)+rng.normal(0,.3,n)
data=np.c_[temp,humidity,pressure]
X=[]; y=[]
for i in range(n-8): X.append(data[i:i+8]); y.append(temp[i+8])
X=np.array(X); y=np.array(y); cut=170
# GRU-like update/reset gated state over three weather variables
def encode(seq):
    h=np.zeros(6)
    for row in seq:
        z=1/(1+np.exp(-np.clip(row.mean()/30,-10,10)))
        r=1/(1+np.exp(-np.clip(row.std()/10,-10,10)))
        candidate=np.tanh(row.mean()/30+r*h)
        h=(1-z)*h+z*candidate
    return h
F=np.array([encode(row) for row in X])
A=np.c_[F[:cut],np.ones(cut)]; w=np.linalg.pinv(A)@y[:cut]
p=np.c_[F[cut:],np.ones(len(F)-cut)]@w
print("Weather inputs: temperature, humidity, atmospheric pressure")
print("Test temperature MAE:",round(metrics(y[cut:],p)[0],3))
print("Forecast next temperature:",round(float(p[-1]),2),"°C")
