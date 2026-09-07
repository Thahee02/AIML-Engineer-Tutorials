import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve the folder where the model and scaler are stored
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# Load the trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

st.title("Student Performance Predictor")
st.write("Enter the student details below to predict whether the student will pass.")

with st.form("student_form"):
    study_hours = st.slider("Study Hours", min_value=1, max_value=10, value=5)
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=90)
    previous_score = st.slider("Previous Score", min_value=0, max_value=100, value=75)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Score": [previous_score]
    })

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0].max()

    if prediction == 1:
        st.success(f"Prediction: PASS")
        st.write(f"Confidence: {probability * 100:.2f}%")
    else:
        st.error(f"Prediction: FAIL")
        st.write(f"Confidence: {probability * 100:.2f}%")

    st.caption("The model was trained using the same feature names and scaler saved during training.")
