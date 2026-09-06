import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)
import joblib

# ==================================================
# 1. CREATE DATASET
# ==================================================

df = pd.DataFrame({
    "Area": [1000, 1200, 1500, 1800, 2200],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Bathrooms": [1, 2, 2, 3, 3],
    "Age": [10, 8, 5, 3, 2],
    "Distance": [8, 6, 5, 4, 3],
    "Price": [15, 19, 24, 30, 38]
})

# ==================================================
# 2. Understand the Dataset
# ==================================================

print("Head of the dataset:")
print(df.head())

print("\nShape of the dataset:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# ==================================================
# 3. CHECK DATA QUALITY
# ==================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==================================================
# 4. EDA
# ==================================================

# 4.1 Area vs Price
sns.scatterplot(
    data=df,
    x="Area",
    y="Price"
)

plt.title("Area vs Price")
plt.show()

# 4.2 Bedrooms vs Price
sns.scatterplot(
    data=df,
    x="Bedrooms",
    y="Price"
)
plt.title("Bedrooms vs Price")
plt.show()

# 4.3 Bathrooms vs Price
sns.scatterplot(
    data=df,
    x="Bathrooms",
    y="Price"
)
plt.title("Bathrooms vs Price")
plt.show()

# 4.4 Age vs Price
sns.scatterplot(
    data=df,
    x="Age",
    y="Price"
)

plt.title("Age vs Price")
plt.show()

# 4.5 Distance vs Price
sns.scatterplot(
    data=df,
    x="Distance",
    y="Price"
)

plt.title("Distance vs Price")
plt.show()

# ==================================================
# 5. CORRELATION
# ==================================================

print("\nCorrelation Matrix:")
correlation = df.corr()
print(correlation)

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Matrix")
plt.show()

# ==================================================
# 6. DEFINE FEATURES and TARGET
# ==================================================

X = df[["Area", "Bedrooms", "Bathrooms", "Age", "Distance"]]
y = df["Price"]

# ==================================================
# 7. SPLIT DATASET
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==================================================
# 8. CREATE MODEL
# ==================================================

model = LinearRegression()

# ==================================================
# 9. TRAIN MODEL
# ==================================================

model.fit(
    X_train,
    y_train
)

# ==================================================
# 10. PREDICTIONS
# ==================================================

predictions = model.predict(
    X_test
)

# ==================================================
# 11. EVALUATE MODEL
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

print("\n========== EVALUATION METRICS ==========")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# ==================================================
# 12. MODEL COEFFICIENTS
# ==================================================

print("\n========== MODEL COEFFICIENTS ==========")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# ==================================================
# 13. ACTUAL VS PREDICTED PLOT
# ==================================================

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Prices")

plt.show()


# ==================================================
# 14. RESIDUALS
# ==================================================

residuals = y_test - predictions

print("\n========== RESIDUALS ==========")

print(residuals)

# ==================================================
# 15. PREDICT NEW DATA
# ==================================================

new_house = pd.DataFrame({
    "Area": [2000],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5], 
    "Distance": [4]
})

new_prediction = model.predict(
    new_house
)
print("\n========== NEW PREDICTION ==========")
print("Predicted Price:", new_prediction[0])

# =================================================
# 16. SAVE MODEL
# ==================================================

joblib.dump(model, "house_price_model.pkl")

print("\nModel saved as 'house_price_model.pkl'")