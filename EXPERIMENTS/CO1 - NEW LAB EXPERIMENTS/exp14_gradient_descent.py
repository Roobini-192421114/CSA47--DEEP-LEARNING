# Minimize f(x) = (x - 5)^2 using gradient descent
def cost(x):
    return (x - 5) ** 2

def gradient(x):
    return 2 * (x - 5)

x = 0.0
learning_rate = 0.1
iterations = 30

print("Iteration 0: x =", x, "cost =", cost(x))
for step in range(1, iterations + 1):
    x = x - learning_rate * gradient(x)
    if step % 5 == 0 or step == 1:
        print(f"Iteration {step}: x={x:.6f}, cost={cost(x):.8f}")

print("\nApproximate minimum at x =", round(x, 6))
print("Minimum cost =", round(cost(x), 8))
