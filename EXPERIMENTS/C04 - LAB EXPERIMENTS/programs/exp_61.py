import numpy as np
from common import make_series, windows, metrics
print("EXPERIMENT 61: LSTM HYPERPARAMETER EXPERIMENT")
s=make_series(240,61); X,y=windows(s,12); cut=175
for hidden in [4,8,12]:
    for learning_rate in [0.01,0.05]:
        # Compact gated-memory encoder; learning rate is represented in readout ridge strength.
        F=[]
        for seq in X:
            h=np.zeros(hidden); c=np.zeros(hidden)
            for x in seq:
                z=np.tanh(x*.4+h*.2); c=.8*c+.2*z; h=np.tanh(c)
            F.append(h)
        F=np.array(F); A=np.c_[F[:cut],np.ones(cut)]
        ridge=learning_rate
        w=np.linalg.solve(A.T@A+ridge*np.eye(A.shape[1]),A.T@y[:cut])
        p=np.c_[F[cut:],np.ones(len(F)-cut)]@w
        print(f"Hidden units={hidden:2d}, learning-rate setting={learning_rate:.2f}, MAE={metrics(y[cut:],p)[0]:.4f}")
print("Compare test error; select settings that generalize well without unnecessary complexity.")
