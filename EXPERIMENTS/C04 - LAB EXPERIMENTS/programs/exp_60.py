import numpy as np
from common import make_series, windows, rnn_features, fit_readout, predict_readout, metrics
print("EXPERIMENT 60: EFFECT OF OBSERVATION WINDOW LENGTH")
s=make_series(260,60)
for length in [5,10,20,30]:
    X,y=windows(s,length); cut=int(.8*len(X))
    F=rnn_features(X,hidden=8,seed=5); w=fit_readout(F[:cut],y[:cut])
    p=predict_readout(F[cut:],w)
    print(f"Window={length:2d} | Test MAE={metrics(y[cut:],p)[0]:.4f}")
print("A suitable window balances useful history and unnecessary sequence length.")
