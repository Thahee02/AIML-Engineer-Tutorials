import numpy as np
from sklearn.linear_model import LinearRegression

# Train data
# Hours Studied
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

# Scores
y = np.array([
    35,
    40,
    50,
    55,
    65,
    70,
    80,
    90
])

# Define the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# get input from user for hours studied
hours = float(input("Enter the number of hours studied: "))

# Predict score for a student who studied for the entered hours
predicted_score = model.predict([[hours]])

# Print the predicted score
print(f"Predicted score for a student who studied for 9 hours: {predicted_score[0]:.2f}")