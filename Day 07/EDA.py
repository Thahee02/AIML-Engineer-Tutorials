import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.DataFrame({
    "Age": [
        22, 25, 28, 30, 35,
        24, 27, 31, 40, 45
    ],

    "Experience": [
        1, 2, 4, 5, 10,
        2, 3, 6, 12, 15
    ],

    "Salary": [
        35000, 40000, 50000, 55000, 90000,
        42000, 48000, 65000, 100000, 120000
    ],

    "Department": [
        "IT", "HR", "IT", "Finance", "IT",
        "HR", "IT", "Finance", "IT", "Finance"
    ]
})


# ==================================
# BASIC INFORMATION
# ==================================

print("Dataset:")
print(df)

print("\nShape:")
print(df.shape)

print("\nInformation:")
df.info()

print("\nStatistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


# ==================================
# AGE DISTRIBUTION
# ==================================

sns.histplot(
    data=df,
    x="Age"
)

plt.title("Age Distribution")
plt.show()


# ==================================
# SALARY DISTRIBUTION
# ==================================

sns.histplot(
    data=df,
    x="Salary"
)

plt.title("Salary Distribution")
plt.show()


# ==================================
# EXPERIENCE VS SALARY
# ==================================

sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary"
)

plt.title("Experience vs Salary")
plt.show()


# ==================================
# DEPARTMENT COUNT
# ==================================

sns.countplot(
    data=df,
    x="Department"
)

plt.title("Employees by Department")
plt.show()


# ==================================
# SALARY BY DEPARTMENT
# ==================================

sns.boxplot(
    data=df,
    x="Department",
    y="Salary"
)

plt.title("Salary by Department")
plt.show()


# ==================================
# CORRELATION
# ==================================

numeric_df = df[
    ["Age", "Experience", "Salary"]
]

correlation = numeric_df.corr()

print("\nCorrelation:")
print(correlation)


# ==================================
# HEATMAP
# ==================================

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()