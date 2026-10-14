
import numpy as np

marks = np.array([75, 90, 65, 95, 80])

print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Minimum:", np.min(marks))
print("Maximum:", np.max(marks))
print("Standard Deviation:", np.std(marks))

topper_index = np.argmax(marks)
print("Topper position:", topper_index)
print("Topper mark:", marks[topper_index])


max_mark = marks[0]
manual_index = 0

for i in range(1, len(marks)):
    if marks[i] > max_mark:
        max_mark = marks[i]
        manual_index = i

print("Manual topper position:", manual_index)

if topper_index == manual_index:
    print("Check passed: Both positions match")
else:
    print("Check failed: Positions do not match")
