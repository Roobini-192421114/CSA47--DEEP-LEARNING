import math

x = list(range(1, 11))
y = [8, 12, 18, 25, 30, 35, 43, 48, 52, 60]

n = len(x)
mx = sum(x) / n
my = sum(y) / n

b = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / sum((v - mx) ** 2 for v in x)
a = my - b * mx

pred = [a + b * v for v in x]
mse = sum((y[i] - pred[i]) ** 2 for i in range(n)) / n

print("Actual Marks\tPredicted Marks")
for i in range(n):
    print(y[i], "\t\t", round(pred[i], 2))

print("Mean Squared Error:", round(mse, 2))
