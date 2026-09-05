import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8],
    "Salary": [
        35000,
        40000,
        45000,
        50000,
        60000,
        65000,
        70000,
        80000
    ],
    "Department": [
        "IT",
        "IT",
        "HR",
        "IT",
        "Finance",
        "HR",
        "IT",
        "Finance"
    ]
})

print(df)


# scatter plot
sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary"
)

plt.title("Experience vs Salary")

plt.show()

# Histogram
sns.histplot(
    data=df,
    x="Salary"
)

plt.title("Salary Distribution")

plt.show()

# count plot
sns.countplot(
    data=df,
    x="Department"
)

plt.title("Employees by Department")

plt.show()

# box plot
sns.boxplot(
    data=df,
    x="Department",
    y="Salary"
)

plt.title("Salary by Department")

plt.show()

# heatmap
correlation = df[
    ["Experience", "Salary"]
].corr()

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Matrix")

plt.show()