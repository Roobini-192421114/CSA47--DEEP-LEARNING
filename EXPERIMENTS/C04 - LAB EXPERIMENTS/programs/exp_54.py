import numpy as np
from common import make_series, windows, rnn_features, fit_readout, predict_readout, metrics
print("EXPERIMENT 54: RNN AND LSTM COMPARISON")
s=make_series(240,54); X,y=windows(s,10); cut=180
# RNN representation
Fr=rnn_features(X,hidden=8,seed=1); wr=fit_readout(Fr[:cut],y[:cut]); pr=predict_readout(Fr[cut:],wr)
# LSTM-style gated memory representation
def lstm_features(data, hidden=8):
    rng=np.random.default_rng(2); out=[]
    for seq in data:
        h=np.zeros(hidden); c=np.zeros(hidden)
        for x in seq:
            z=np.tanh(x*0.4+h*0.2); gate=1/(1+np.exp(-x))
            c=gate*c+(1-gate)*z; h=np.tanh(c)
        out.append(h)
    return np.array(out)
Fl=lstm_features(X); wl=fit_readout(Fl[:cut],y[:cut]); pl=predict_readout(Fl[cut:],wl)
print("RNN MAE:",round(metrics(y[cut:],pr)[0],4))
print("LSTM-style MAE:",round(metrics(y[cut:],pl)[0],4))
print("LSTM uses gated cell memory to retain useful information.")
