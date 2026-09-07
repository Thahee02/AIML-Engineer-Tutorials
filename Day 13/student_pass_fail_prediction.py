import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==================================
# CREATE DATASET
# ==================================

data = {
    "Study_Hours": [
        1, 2, 2, 3, 3,
        4, 5, 5, 6, 7,
        7, 8, 9, 10, 10
    ],

    "Attendance": [
        50, 55, 60, 62, 65,
        68, 70, 72, 75, 80,
        82, 85, 90, 92, 95
    ],

    "Pass": [
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}


df = pd.DataFrame(data)


# ==================================
# FEATURES AND TARGET
# ==================================

X = df[
    [
        "Study_Hours",
        "Attendance"
    ]
]

y = df["Pass"]


# ==================================
# TRAIN TEST SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================
# CREATE MODEL
# ==================================

model = LogisticRegression()


# ==================================
# TRAIN
# ==================================

model.fit(
    X_train,
    y_train
)


# ==================================
# PREDICTION
# ==================================

y_pred = model.predict(
    X_test
)


# ==================================
# EVALUATION
# ==================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ==================================
# RESULTS
# ==================================

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(cm)


# ==================================
# CLASSIFICATION REPORT
# ==================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)