import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib

# Create a sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "Previous_Score": [40, 45, 50, 55, 60, 65, 70, 75, 80, 90],
    "Pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)
print(df)

# Specify the features and target variable
X = df[["Study_Hours", "Attendance", "Previous_Score"]]
y = df["Pass"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find the optimal number of neighbors (k) using the elbow method
k_values = range(1, 8)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plot the elbow curve
plt.plot(k_values, accuracies)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Elbow Method for Optimal k")
plt.show()

print("Accuracies for different k values:")
for k, acc in zip(k_values, accuracies):
    print(f"k={k}: Accuracy={acc:.2f}")

# Train a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=6) # Using k=6 based on the elbow method

# Fit the model to the training data
knn.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Print the classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model and scaler to disk
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as 'knn_model.pkl'")