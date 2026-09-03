import numpy as np

# Maths, Science and English marks of 5 students
marks = np.array([
    [80, 75, 90],
    [65, 70, 60],
    [90, 95, 85],
    [55, 60, 58],
    [75, 80, 70]
])

# Number of students and subjects
print(f"Number of students: {marks.shape[0]}")
print(f"Number of subjects: {marks.shape[1]}")

# Average for each student
print(f"Average for each student: {np.mean(marks, axis=1)}")

# Average mark of each subject
print(f"Average mark of each subject: {np.mean(marks, axis=0)}")

# Highest marks
print(f"Highest marks: {np.max(marks)}")

# Lowest mark
print(f"Lowest mark: {np.min(marks)}")

# Overall average
print(f"Overall average: {np.mean(marks)}")