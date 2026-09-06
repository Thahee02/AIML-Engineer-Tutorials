import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ==========================================
# LOAD DATA
# ==========================================

df = pd.DataFrame({
    "Area": [
        1000, 1200, 1500, 1800, 2000,
        2200, 2500, 2800, 3000, 3500,
        1200, 1500, 1800, 2000, 5000
    ],

    "Bedrooms": [
        2, 2, 3, 3, 3,
        4, 4, 4, 5, 5,
        2, 3, 3, 4, 8
    ],

    "Bathrooms": [
        1, 2, 2, 2, 3,
        3, 3, 4, 4, 5,
        2, 2, 3, 3, 7
    ],

    "Age": [
        15, 10, 8, 5, 6,
        4, 3, 2, 1, 1,
        12, 9, 7, 5, 2
    ],

    "Location": [
        "Colombo", "Colombo", "Kandy", "Colombo",
        "Galle", "Colombo", "Kandy", "Colombo",
        "Galle", "Colombo", "Kandy", "Colombo",
        "Galle", "Kandy", "Colombo"
    ],

    "Price": [
        12000000, 15000000, 18000000, 22000000,
        25000000, 28000000, 32000000, 36000000,
        40000000, 45000000, 15000000, 19000000,
        23000000, 27000000, 70000000
    ]
})


# ==========================================
# BASIC INFORMATION
# ==========================================

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nInformation:")
df.info()

print("\nStatistics:")
print(df.describe())


# ==========================================
# DATA QUALITY
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())


# ==========================================
# CATEGORICAL ANALYSIS
# ==========================================

print("\nLocation Counts:")
print(df["Location"].value_counts())

print("\nUnique Locations:")
print(df["Location"].unique())


# ==========================================
# TARGET ANALYSIS
# ==========================================

print("\nPrice Statistics:")
print(df["Price"].describe())

print("\nPrice Skewness:")
print(df["Price"].skew())


# ==========================================
# PRICE DISTRIBUTION
# ==========================================

sns.histplot(
    data=df,
    x="Price"
)

plt.title("Price Distribution")
plt.show()


# ==========================================
# AREA VS PRICE
# ==========================================

sns.scatterplot(
    data=df,
    x="Area",
    y="Price"
)

plt.title("Area vs Price")
plt.show()


# ==========================================
# AGE VS PRICE
# ==========================================

sns.scatterplot(
    data=df,
    x="Age",
    y="Price"
)

plt.title("Age vs Price")
plt.show()


# ==========================================
# BEDROOMS VS PRICE
# ==========================================

sns.boxplot(
    data=df,
    x="Bedrooms",
    y="Price"
)

plt.title("Bedrooms vs Price")
plt.show()


# ==========================================
# LOCATION VS PRICE
# ==========================================

sns.boxplot(
    data=df,
    x="Location",
    y="Price"
)

plt.title("Location vs Price")
plt.show()


# ==========================================
# CORRELATION
# ==========================================

numeric_columns = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "Price"
]

correlation = df[
    numeric_columns
].corr()

print("\nCorrelation Matrix:")
print(correlation)


# ==========================================
# CORRELATION HEATMAP
# ==========================================

sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Heatmap")
plt.show()


# ==========================================
# CORRELATION WITH TARGET
# ==========================================

print("\nCorrelation with Price:")

print(
    correlation["Price"].sort_values(
        ascending=False
    )
)