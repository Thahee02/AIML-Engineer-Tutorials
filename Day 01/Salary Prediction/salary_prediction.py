import numpy as np
from sklearn.linear_model import LinearRegression

# Training Data
# Years of Experience (X) and Salary (Y)
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])
y = np.array([
    30000,
    40000,
    50000,
    60000,
    70000
])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict salary for a someone with 6 years of experience
predicted_salary = model.predict([[6]])

# Print the predicted salary
print(f"Predicted salary for 6 years of experience: ${predicted_salary[0]:,.2f}")