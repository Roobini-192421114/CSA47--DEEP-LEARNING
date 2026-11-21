import numpy as np
print("EXPERIMENT 59: VANISHING GRADIENT IN A SIMPLE RNN")
for steps in [5,10,20,40,60]:
    # Repeated derivative through tanh with a small recurrent weight.
    grad=1.0
    for _ in range(steps): grad*=0.5
    print(f"Sequence steps={steps:2d} | gradient magnitude={grad:.8f}")
print("The gradient becomes very small over long sequences, making early time steps hard to learn.")
print("LSTM/GRU gates help preserve information and improve gradient flow.")
