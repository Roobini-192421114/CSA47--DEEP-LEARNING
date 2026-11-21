import numpy as np
from common import make_series, windows, rnn_features, fit_readout, predict_readout, metrics
print("EXPERIMENT 55: STOCK PRICE TREND PREDICTION USING LSTM")
price=100+8*make_series(260,55)
X,y=windows(price,15); cut=int(.8*len(X))
# Simple gated-memory sequence representation
def lstm_encode(seq):
    h=0.0; c=0.0
    for x in seq:
        z=np.tanh((x-100)*0.08+h*0.25); f=0.75; i=0.25
        c=f*c+i*z; h=np.tanh(c)
    return h
F=np.array([[lstm_encode(row)] for row in X])
w=fit_readout(F[:cut],y[:cut]); p=predict_readout(F[cut:],w)
mae,mse,rmse=metrics(y[cut:],p)
print(f"Test MAE: {mae:.3f}; MSE: {mse:.3f}; RMSE: {rmse:.3f}")
print(f"Last actual price: {y[-1]:.2f}; predicted: {p[-1]:.2f}")
print("Synthetic stock-price series only; not financial advice or a real market forecast.")
