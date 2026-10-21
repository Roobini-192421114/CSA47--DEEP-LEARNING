import math

Y = [1, 0, 0]
P = [0.7, 0.2, 0.1]
classes = ["Cat", "Dog", "Bird"]

loss = 0

for i in range(3):
    if Y[i] == 1:
        loss -= math.log(P[i])

print("Actual and Predicted Probabilities:")
for i in range(3):
    print(classes[i], "Actual:", Y[i],
          "Predicted:", P[i])

print("Cross-Entropy Loss:", round(loss, 4))
