import numpy as np
import matplotlib.pyplot as plt

# Minimize f(w) = (w - 4)^2
def run_gradient_descent(learning_rate, steps=30):
    w = -5.0
    losses = []
    for _ in range(steps):
        loss = (w - 4) ** 2
        losses.append(loss)
        gradient = 2 * (w - 4)
        w -= learning_rate * gradient
    return w, losses

rates = [0.01, 0.1, 0.5, 1.1]
for rate in rates:
    final_w, losses = run_gradient_descent(rate)
    print(f"Learning rate {rate}: final w={final_w:.5f}, final loss={losses[-1]:.6f}")
    plt.plot(losses, label=f"lr={rate}")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Effect of Learning Rate on Gradient Descent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nA small rate learns slowly; a suitable rate converges faster; a very large rate may oscillate or diverge.")
