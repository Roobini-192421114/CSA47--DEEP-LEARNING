x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [6, 15, 25, 39, 55, 75, 96, 122, 150, 183]

print("House Area\tHouse Price")
for i in range(len(x)):
    print(x[i], "\t\t", y[i])

print("\nDegree 1: Underfitting")
print("Degree 2: Better fit")
print("Degree 10: Overfitting")
