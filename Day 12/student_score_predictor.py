import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================
# DATA
# ==============================

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


# ==============================
# FEATURES AND TARGET
# ==============================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignments"
    ]
]

y = df["Final_Score"]


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==============================
# MODEL
# ==============================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==============================
# TRAIN PREDICTIONS
# ==============================

train_predictions = model.predict(
    X_train
)


# ==============================
# TEST PREDICTIONS
# ==============================

test_predictions = model.predict(
    X_test
)


# ==============================
# TRAIN METRICS
# ==============================

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_predictions
    )
)

train_r2 = r2_score(
    y_train,
    train_predictions
)


# ==============================
# TEST METRICS
# ==============================

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


# ==============================
# RESULTS
# ==============================

print("\n========== MODEL PERFORMANCE ==========")

print("\nTraining:")
print("MAE :", train_mae)
print("RMSE:", train_rmse)
print("R²  :", train_r2)

print("\nTesting:")
print("MAE :", test_mae)
print("RMSE:", test_rmse)
print("R²  :", test_r2)


# ==============================
# ACTUAL VS PREDICTED
# ==============================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": test_predictions
})

print("\n========== PREDICTIONS ==========")

print(results)


# ==============================
# RESIDUALS
# ==============================

residuals = y_test.values - test_predictions

print("\n========== RESIDUALS ==========")

print(residuals)