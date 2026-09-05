import matplotlib.pyplot as plt

months = [
    "January",
    "February",
    "March",
    "April",
    "May"
]

sales = [
    100,
    120,
    150,
    140,
    180
]

plt.plot(months, sales)

plt.title("Monthly Sales")

plt.xlabel("Month")
plt.ylabel("Sales")

plt.show()