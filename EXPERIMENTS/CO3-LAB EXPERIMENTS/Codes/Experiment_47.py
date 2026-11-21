# Experiment 47: RNN-inspired sequential sensor data demo
# Autoregression baseline using lagged sensor readings.
import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
rng=np.random.default_rng(47); t=np.arange(300); signal=np.sin(t/15)+.15*rng.normal(size=300)
X=np.array([signal[i-5:i] for i in range(5,250)]); y=signal[5:250]
m=LinearRegression().fit(X,y); pred=m.predict(X)
print('Lagged sensor prediction R2:',m.score(X,y))
plt.figure(figsize=(7,4)); plt.plot(y[:100],label='Observed'); plt.plot(pred[:100],label='Predicted'); plt.xlabel('Sample'); plt.ylabel('Sensor reading'); plt.title('Sequence prediction demo'); plt.legend(); plt.tight_layout(); plt.savefig('../Outputs/47_sensor_sequence.png')
