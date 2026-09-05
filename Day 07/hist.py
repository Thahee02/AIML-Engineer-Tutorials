import matplotlib.pyplot as plt

ages = [
    21, 22, 23, 24, 25,
    26, 27, 28, 29, 30,
    31, 32, 35, 40, 45
]

plt.hist(ages)

plt.title("Age Distribution")

plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()