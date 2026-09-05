import pandas as pd

employees = {
    "Name": ["Alex","John","Sarah","David","Mike","Emma","Daniel"],
    "Age": [23,27,25,32,29,24,35],
    "Department": ["IT","IT","HR","Finance","IT","HR","Finance"],
    "Salary": [50000,65000,55000,70000,80000,52000,90000]
}


# Create DataFrame
df = pd.DataFrame(employees)

print("========== EMPLOYEE DATA ==========")
print(df)

# Number of employees
print("\nNumber of employees:")
print(len(df))

# Average age
print("\nAverage age:")
print(df["Age"].mean())

# Average salary
print("\nAverage salary:")
print(df["Salary"].mean())

# Highest salary
print("\nHighest salary:")
print(df["Salary"].max())

# Lowest salary
print("\nLowest salary:")
print(df["Salary"].min())

# Employee with highest salary
print("\nEmployee with highest salary:")
highest_salary_employee = df.loc[
    df["Salary"].idxmax()
]
print(highest_salary_employee)

# Employees earning more than 60,000
print("\nEmployees earning more than 60,000:")
print(df[df["Salary"] > 60000])

# Average salary by department
print("\nAverage salary by department:")
print(
    df.groupby("Department")["Salary"].mean()
)

# Number of employees by department
print("\nEmployees by department:")
print(
    df["Department"].value_counts()
)

# Sort by salary
print("\nEmployees sorted by salary:")
print(
    df.sort_values(
        "Salary",
        ascending=False
    )
)