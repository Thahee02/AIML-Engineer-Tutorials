import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CREATE DATASET
# ==========================================

data = {
    "Area": [
        1200,
        1500,
        1800,
        np.nan,
        2200,
        1500
    ],

    "Bedrooms": [
        2,
        3,
        3,
        4,
        np.nan,
        3
    ],

    "Age": [
        10,
        5,
        8,
        3,
        2,
        5
    ],

    "City": [
        "Colombo",
        "Kandy",
        "Colombo",
        "Galle",
        "Colombo",
        "Kandy"
    ],

    "Price": [
        15000000,
        20000000,
        25000000,
        30000000,
        35000000,
        20000000
    ]
}


df = pd.DataFrame(data)


# ==========================================
# 2. DISPLAY DATA
# ==========================================

print("Original Dataset:")
print(df)


# ==========================================
# 3. DATA INFORMATION
# ==========================================

print("\nDataset Information:")
df.info()


# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 5. FILL MISSING VALUES
# ==========================================

df["Area"] = df["Area"].fillna(
    df["Area"].median()
)

df["Bedrooms"] = df["Bedrooms"].fillna(
    df["Bedrooms"].median()
)


# ==========================================
# 6. CHECK DUPLICATES
# ==========================================

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ==========================================
# 7. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()


# ==========================================
# 8. ENCODE CITY
# ==========================================

df = pd.get_dummies(
    df,
    columns=["City"],
    dtype=int
)


print("\nAfter Preprocessing:")
print(df)


# ==========================================
# 9. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Price", axis=1)

y = df["Price"]


print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)


# ==========================================
# 10. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 11. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ==========================================
# 12. FINAL RESULTS
# ==========================================

print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

print("\nTraining Target:")
print(y_train)

print("\nTesting Target:")
print(y_test)