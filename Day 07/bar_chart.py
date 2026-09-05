import matplotlib.pyplot as plt

departments = [
    "IT",
    "HR",
    "Finance",
    "Marketing"
]

employees = [
    50,
    20,
    30,
    40
]

plt.bar(departments, employees)

plt.title("Employees by Department")

plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.show()