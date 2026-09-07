import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Specify the features variable
X = df[["Study_Hours", "Attendance", "Previous_Score"]]

# Specify the target variable
y = df["Pass"] # 0 = Fail, 1 = Pass

# Split the dataset into training and testing sets
# stratify=y ensures that the class distribution is preserved in both training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# dynamic max_depth based on the number of features
for depth in [1, 2, 3, 4, 5, None]:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(
        depth,
        train_accuracy,
        test_accuracy
    )

# Train a Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)

# Fit the model to the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# feature_importances
print("\nFeature Importances:")
for feature, importance in zip(X.columns, dt_classifier.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# visualize the decision tree
plt.figure(figsize=(12, 8))

plot_tree(
    dt_classifier,
    feature_names=X.columns,
    class_names=["Fail", "Pass"],
    filled=True
)

plt.show()


# Save the trained model to disk
joblib.dump(dt_classifier, "dt_model.pkl")