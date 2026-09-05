import matplotlib.pyplot as plt

experience = [1, 2, 3, 4, 5, 6]
salary = [35000, 40000, 45000, 52000, 60000, 65000]

plt.scatter(experience, salary)

plt.title("Experience vs Salary")

plt.xlabel("Experience")
plt.ylabel("Salary")

plt.show()