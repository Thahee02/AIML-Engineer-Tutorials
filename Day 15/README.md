# Student Performance Prediction

This project predicts whether a student will pass based on three features:

- Study Hours
- Attendance
- Previous Score

It uses a K-Nearest Neighbors (KNN) classifier and a StandardScaler for preprocessing.

## Project Files

- `student_performance_classifier.py` - trains the model and saves the artifacts
- `prediction.py` - runs a sample prediction from saved model files
- `app.py` - Streamlit web interface for user input
- `knn_model.pkl` - trained KNN model
- `scaler.pkl` - fitted scaler used during training

## Dependencies

Install required libraries:

```bash
pip install pandas matplotlib scikit-learn joblib streamlit
```

## Train the Model

Run:

```bash
python student_performance_classifier.py
```

This script:

- creates a sample dataset
- splits data into train/test sets
- standardizes features
- trains a KNN model
- evaluates accuracy and classification metrics
- saves the model and scaler as pickle files

## Run a Simple Prediction

```bash
python prediction.py
```

This loads the saved model and scaler, passes a sample student record, and prints whether the student is predicted to pass or fail.

## Run the Streamlit App

```bash
python -m streamlit run app.py
```

Then open the browser URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Expected Output

The model predicts a student outcome based on:

- hours studied
- attendance percentage
- previous score

A sample prediction may look like:

```text
The student is predicted to pass.
```

## Notes

- The `scaler.pkl` file must be saved and loaded with the same feature names used in training.
- The feature names used in the app must match the training data exactly.
