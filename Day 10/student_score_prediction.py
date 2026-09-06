import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)


# ==================================================
# 1. CREATE DATASET
# ==================================================

df = pd.DataFrame({
    "Study_Hours": [
        1, 2, 2, 3, 3,
        4, 4, 5, 5, 6,
        6, 7, 7, 8, 9,
        10, 2, 4, 6, 8
    ],

    "Attendance": [
        60, 65, 70, 72, 75,
        78, 80, 82, 85, 87,
        88, 90, 91, 93, 95,
        98, 68, 79, 89, 94
    ],

    "Assignments": [
        50, 55, 60, 62, 65,
        68, 70, 75, 78, 80,
        82, 85, 87, 90, 93,
        97, 58, 72, 84, 91
    ],

    "Final_Score": [
        45, 50, 53, 57, 60,
        64, 67, 72, 75, 79,
        82, 85, 87, 90, 94,
        98, 52, 66, 81, 92
    ]
})


# ==================================================
# 2. UNDERSTAND DATA
# ==================================================

print("\n========== DATASET ==========")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nStatistics:")
print(df.describe())


# ==================================================
# 3. CHECK DATA QUALITY
# ==================================================

print("\n========== DATA QUALITY ==========")

print("Missing values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())


# ==================================================
# 4. EDA
# ==================================================

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Score"
)

plt.title("Study Hours vs Final Score")
plt.show()


sns.scatterplot(
    data=df,
    x="Attendance",
    y="Final_Score"
)

plt.title("Attendance vs Final Score")
plt.show()


sns.scatterplot(
    data=df,
    x="Assignments",
    y="Final_Score"
)

plt.title("Assignments vs Final Score")
plt.show()


# ==================================================
# 5. CORRELATION
# ==================================================

print("\n========== CORRELATION ==========")

correlation = df.corr()

print(correlation)

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Matrix")
plt.show()


# ==================================================
# 6. DEFINE FEATURES
# ==================================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignments"
    ]
]


# ==================================================
# 7. DEFINE TARGET
# ==================================================

y = df["Final_Score"]


# ==================================================
# 8. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==================================================
# 9. CREATE MODEL
# ==================================================

model = LinearRegression()


# ==================================================
# 10. TRAIN MODEL
# ==================================================

model.fit(
    X_train,
    y_train
)


# ==================================================
# 11. PREDICTIONS
# ==================================================

predictions = model.predict(
    X_test
)


# ==================================================
# 12. ACTUAL VS PREDICTED
# ==================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n========== PREDICTIONS ==========")

print(results)


# ==================================================
# 13. EVALUATION
# ==================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = root_mean_squared_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n========== MODEL EVALUATION ==========")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# ==================================================
# 14. MODEL COEFFICIENTS
# ==================================================

print("\n========== MODEL PARAMETERS ==========")

print("Coefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)


# ==================================================
# 15. ACTUAL VS PREDICTED PLOT
# ==================================================

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")

plt.title("Actual vs Predicted Scores")

plt.show()


# ==================================================
# 16. RESIDUALS
# ==================================================

residuals = y_test - predictions

print("\n========== RESIDUALS ==========")

print(residuals)


# ==================================================
# 17. PREDICT NEW STUDENT
# ==================================================

new_student = pd.DataFrame({
    "Study_Hours": [7],
    "Attendance": [92],
    "Assignments": [88]
})

new_prediction = model.predict(
    new_student
)

print("\n========== NEW STUDENT ==========")

print(
    "Predicted Final Score:",
    new_prediction[0]
)


# ==================================================
# 18. SAVE MODEL
# ==================================================

joblib.dump(
    model,
    "student_score_model.pkl"
)

print("\nModel saved successfully!")