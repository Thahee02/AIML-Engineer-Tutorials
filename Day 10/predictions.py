import joblib
import pandas as pd

model = joblib.load("student_score_model.pkl")

study_hours = float(input("\nEnter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))
assignments = float(input("Enter Assignments Score: "))

new_student = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Assignments": [assignments]
})

predictions = model.predict(new_student)

print(
    "Predicted Final Score:", predictions[0]
)