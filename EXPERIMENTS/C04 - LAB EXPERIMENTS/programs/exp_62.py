import numpy as np
from common import make_series, windows, rnn_features, fit_readout, predict_readout, metrics
print("EXPERIMENT 62: RNN ELECTRICITY FORECASTING METRICS")
s=make_series(250,62); X,y=windows(s,10); cut=180
F=rnn_features(X,hidden=8,seed=6); w=fit_readout(F[:cut],y[:cut]); p=predict_readout(F[cut:],w)
mae,mse,rmse=metrics(y[cut:],p)
print(f"MAE  = {mae:.5f}")
print(f"MSE  = {mse:.5f}")
print(f"RMSE = {rmse:.5f}")
print("Lower values indicate predictions closer to actual consumption.")
