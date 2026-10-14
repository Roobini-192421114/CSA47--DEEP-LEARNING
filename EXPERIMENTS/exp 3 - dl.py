
import numpy as np


steps = np.array([
    5000, 6200, 7100, 4500, 8000,
    9200, 6800, 7500, 8300, 6000,
    7200, 9000, 5500, 7800, 8500,
    6700, 9400, 8100, 7300, 6500,
    8800, 9600, 7000, 5900, 8200,
    9100, 7600, 8400, 6900, 10000
])


first_week = steps[:7]


last_five = steps[-5:]


alternate_days = steps[::2]


reversed_steps = steps[::-1]


print("First week:", first_week)
print("Last 5 days:", last_five)
print("Every alternate day:", alternate_days)
print("Reversed vector:", reversed_steps)


print("\nLength checks:")
print("First week:", len(first_week))
print("Last 5 days:", len(last_five))
print("Alternate days:", len(alternate_days))
print("Reversed vector:", len(reversed_steps))
