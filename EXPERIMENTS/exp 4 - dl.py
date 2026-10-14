
import numpy as np


week1 = np.array([100, 200, 150, 300, 250, 180, 400])


week2 = np.array([120, 180, 170, 250, 280, 160, 450])


total = week1 + week2


change = week2 - week1

print("Week 1 Sales:", week1)
print("Week 2 Sales:", week2)
print("Total Sales:", total)
print("Change in Sales:", change)


print("Products whose sales fell:")

for i in range(len(change)):
    if change[i] < 0:
        print("Product", i + 1)
