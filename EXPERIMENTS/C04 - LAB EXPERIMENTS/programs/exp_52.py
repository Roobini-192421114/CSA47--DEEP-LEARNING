import numpy as np
from common import make_series, windows, rnn_features, fit_readout, predict_readout, metrics
print("EXPERIMENT 52: ELECTRICITY CONSUMPTION FORECASTING WITH RNN")
series=make_series(260,seed=52); X,y=windows(series,12); split=int(.8*len(X))
F=rnn_features(X,hidden=10,seed=3); w=fit_readout(F[:split],y[:split])
pred=predict_readout(F[split:],w); mae,mse,rmse=metrics(y[split:],pred)
print(f"Training samples: {split}; Testing samples: {len(y)-split}")
print(f"MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}")
print(f"Next predicted normalized demand: {pred[-1]:.3f}")
print("Synthetic time-series demonstration; replace with actual electricity data for a real forecast.")
