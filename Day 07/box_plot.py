import matplotlib.pyplot as plt

salary = [
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    500000
]

plt.boxplot(salary)

plt.title("Salary Distribution")

plt.ylabel("Salary")

plt.show()