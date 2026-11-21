import numpy as np
print("EXPERIMENT 57: COMPARISON OF RNN, LSTM AND GRU")
sequence=[0.2,0.4,0.1,0.8,0.7,0.3,0.9]
# RNN state
h=0.0
for x in sequence: h=np.tanh(.7*x+.6*h)
rnn=h
# LSTM gated cell
h=0.0; c=0.0
for x in sequence:
    z=np.tanh(.7*x+.2*h); c=.8*c+.2*z; h=np.tanh(c)
lstm=h
# GRU update/reset gates
h=0.0
for x in sequence:
    z=1/(1+np.exp(-x)); r=1/(1+np.exp(-.5*x))
    candidate=np.tanh(x+r*h); h=(1-z)*h+z*candidate
gru=h
for name,value in [("RNN",rnn),("LSTM",lstm),("GRU",gru)]:
    print(f"{name} final hidden state: {value:.4f}")
print("RNN: simple state; LSTM: separate cell memory and gates; GRU: update and reset gates.")
