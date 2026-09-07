import joblib
import pandas as pd

# Load the trained KNN model and scaler from disk
knn_model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

# New student data for prediction
new_student_data = pd.DataFrame({
    "Study_Hours": [5],
    "Attendance": [75],
    "Previous_Score": [50]
})

# Standardize the new student data using the same fitted scaler used during training
new_student_data_scaled = scaler.transform(new_student_data)

# Make predictions using the loaded model
y_pred_loaded = knn_model.predict(new_student_data_scaled)

# Output the prediction result
if y_pred_loaded[0] == 1:
    print("The student is predicted to pass.")
else:
    print("The student is predicted to fail.")