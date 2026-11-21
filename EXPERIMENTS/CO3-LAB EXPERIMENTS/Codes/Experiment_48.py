# Experiment 48: RNN-style temperature forecasting demonstration
import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
rng=np.random.default_rng(48); n=240; t=np.arange(n); temp=25+4*np.sin(t/18)+.3*rng.normal(size=n)
X=np.array([temp[i-10:i] for i in range(10,n)]); y=temp[10:]
split=int(.8*len(X)); m=LinearRegression().fit(X[:split],y[:split]); pred=m.predict(X[split:])
print('Forecast MAE:',np.mean(np.abs(pred-y[split:])))
plt.figure(figsize=(7,4)); plt.plot(y[split:],label='Actual'); plt.plot(pred,label='Forecast'); plt.xlabel('Time step'); plt.ylabel('Temperature'); plt.title('Temperature forecast demo'); plt.legend(); plt.tight_layout(); plt.savefig('../Outputs/48_temperature_forecast.png')
