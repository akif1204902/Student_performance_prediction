# Student Performance Prediction
# Author: Akif
# Description: Predicts student final scores using machine learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
# Example dataset columns:
# study_hours, attendance, previous_score, final_score
data = pd.read_csv("student_data.csv")

# Features and target
X = data.drop("final_score", axis=1)
y = data["final_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
