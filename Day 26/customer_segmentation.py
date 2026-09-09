# ============================================================
# DAY 26 MINI PROJECT
# Customer Segmentation using K-Means Clustering
# ============================================================

# ------------------------------------------------------------
# 1. Import Libraries
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# 2. Create Dataset
# ------------------------------------------------------------

df = pd.DataFrame({
    "Customer_ID": range(1, 16),

    "Age": [
        20, 22, 25, 27, 30,
        35, 38, 40, 42, 45,
        50, 52, 55, 58, 60
    ],

    "Annual_Income": [
        20, 22, 25, 28, 30,
        40, 42, 45, 48, 50,
        60, 65, 70, 75, 80
    ],

    "Spending_Score": [
        85, 90, 80, 88, 82,
        65, 60, 70, 55, 62,
        30, 25, 35, 20, 15
    ]
})


# ------------------------------------------------------------
# 3. Display Dataset
# ------------------------------------------------------------

print("=" * 60)
print("CUSTOMER DATASET")
print("=" * 60)

print(df)


# ------------------------------------------------------------
# 4. Basic Dataset Information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())


print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ------------------------------------------------------------
# 5. Check Missing Values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# ------------------------------------------------------------
# 6. Select Features for Clustering
# ------------------------------------------------------------

features = [
    "Age",
    "Annual_Income",
    "Spending_Score"
]

X = df[features]


print("\n" + "=" * 60)
print("FEATURES USED FOR CLUSTERING")
print("=" * 60)

print(X)


# ------------------------------------------------------------
# 7. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


print("\n" + "=" * 60)
print("SCALED FEATURES")
print("=" * 60)

print(X_scaled)


# ------------------------------------------------------------
# 8. Elbow Method
# ------------------------------------------------------------

inertias = []

K_range = range(1, 11)

for k in K_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia__)


# ------------------------------------------------------------
# 9. Plot Elbow Curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    K_range,
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.xticks(K_range)

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 10. Silhouette Score for Different K Values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SILHOUETTE SCORES")
print("=" * 60)

silhouette_scores = {}

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )


# ------------------------------------------------------------
# 11. Find Best K Based on Silhouette Score
# ------------------------------------------------------------

best_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

print("\n" + "=" * 60)
print("BEST K")
print("=" * 60)

print("Best K based on silhouette score:", best_k)


# ------------------------------------------------------------
# 12. Train Final K-Means Model
# ------------------------------------------------------------

model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(X_scaled)


# ------------------------------------------------------------
# 13. Display Cluster Assignments
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CUSTOMERS WITH CLUSTER ASSIGNMENTS")
print("=" * 60)

print(df)


# ------------------------------------------------------------
# 14. Get Cluster Centers
# ------------------------------------------------------------

cluster_centers_scaled = model.cluster_centers_


print("\n" + "=" * 60)
print("CLUSTER CENTERS - SCALED")
print("=" * 60)

print(cluster_centers_scaled)


# ------------------------------------------------------------
# 15. Convert Centroids Back to Original Scale
# ------------------------------------------------------------

cluster_centers_original = scaler.inverse_transform(
    cluster_centers_scaled
)

centers_df = pd.DataFrame(
    cluster_centers_original,
    columns=features
)

print("\n" + "=" * 60)
print("CLUSTER CENTERS - ORIGINAL SCALE")
print("=" * 60)

print(centers_df)


# ------------------------------------------------------------
# 16. Analyze Each Cluster
# ------------------------------------------------------------

cluster_summary = df.groupby("Cluster")[features].mean()

print("\n" + "=" * 60)
print("CLUSTER SUMMARY")
print("=" * 60)

print(cluster_summary)


# ------------------------------------------------------------
# 17. Number of Customers in Each Cluster
# ------------------------------------------------------------

cluster_counts = df["Cluster"].value_counts().sort_index()

print("\n" + "=" * 60)
print("CUSTOMERS PER CLUSTER")
print("=" * 60)

print(cluster_counts)


# ------------------------------------------------------------
# 18. Visualize Clusters
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    s=100
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 19. Visualize Cluster Centers
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=df["Cluster"],
    s=100
)

# Convert centroid values to original scale
centers = scaler.inverse_transform(
    model.cluster_centers_
)

plt.scatter(
    centers[:, 1],
    centers[:, 2],
    marker="X",
    s=300
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation with Cluster Centers")

plt.grid(True)

plt.show()


# ------------------------------------------------------------
# 20. Print Individual Customers by Cluster
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CUSTOMERS BY CLUSTER")
print("=" * 60)

for cluster in sorted(df["Cluster"].unique()):

    print(f"\nCluster {cluster}")

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    print(
        cluster_data[
            [
                "Customer_ID",
                "Age",
                "Annual_Income",
                "Spending_Score"
            ]
        ]
    )


# ------------------------------------------------------------
# 21. Create Business Interpretation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BUSINESS INTERPRETATION")
print("=" * 60)

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    avg_age = cluster_data["Age"].mean()
    avg_income = cluster_data["Annual_Income"].mean()
    avg_spending = cluster_data["Spending_Score"].mean()

    print(f"\nCluster {cluster}")

    print(f"Average Age: {avg_age:.1f}")
    print(f"Average Income: {avg_income:.1f}")
    print(f"Average Spending Score: {avg_spending:.1f}")

    if avg_income >= 50 and avg_spending >= 60:

        print("Type: High-Value Customers")
        print("Strategy: Premium products and loyalty rewards")

    elif avg_income >= 50 and avg_spending < 60:

        print("Type: High-Income Low-Spending Customers")
        print("Strategy: Personalized offers and promotions")

    elif avg_income < 50 and avg_spending >= 60:

        print("Type: High-Spending Customers")
        print("Strategy: Loyalty programs and repeat-purchase campaigns")

    else:

        print("Type: Low-Engagement Customers")
        print("Strategy: Discounts and re-engagement campaigns")


# ------------------------------------------------------------
# 22. Final Model Information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL INFORMATION")
print("=" * 60)

print("Number of clusters:", best_k)
print("Inertia:", model.inertia_)

final_silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print(
    "Silhouette Score:",
    round(final_silhouette, 4)
)


# ------------------------------------------------------------
# 23. Final Dataset
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print(df)