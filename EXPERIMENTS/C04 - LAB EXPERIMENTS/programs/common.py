import numpy as np

def make_series(n=240, seed=7):
    rng=np.random.default_rng(seed)
    t=np.arange(n)
    return (0.55*np.sin(t/8)+0.25*np.sin(t/21)+0.002*t+rng.normal(0,0.04,n)).astype(float)

def windows(series, length):
    X=[]; y=[]
    for i in range(len(series)-length):
        X.append(series[i:i+length]); y.append(series[i+length])
    return np.asarray(X), np.asarray(y)

def rnn_features(sequences, hidden=8, seed=1):
    rng=np.random.default_rng(seed)
    W=rng.normal(0,0.35,(hidden,1))
    U=rng.normal(0,0.25,(hidden,hidden))
    b=np.zeros(hidden)
    result=[]
    for seq in sequences:
        h=np.zeros(hidden)
        for x in seq:
            h=np.tanh(W[:,0]*x+U@h+b)
        result.append(h)
    return np.asarray(result)

def fit_readout(features, targets):
    X=np.c_[features, np.ones(len(features))]
    return np.linalg.pinv(X)@targets

def predict_readout(features, weights):
    return np.c_[features,np.ones(len(features))]@weights

def metrics(y, p):
    e=np.asarray(y)-np.asarray(p)
    return float(np.mean(np.abs(e))), float(np.mean(e**2)), float(np.sqrt(np.mean(e**2)))

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-30,30)))
