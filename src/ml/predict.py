from sklearn.metrics import accuracy_score, classification_report
from joblib import load
import pandas as pd

def predict(input_data):
    # Load the trained model
    model = load("../models/random_forest.joblib")

    # Ensure input data matches the training format
    # feature_columns = ["Initials", "Hazard Classification", "Lab Number", "Group"]
    input_df = pd.DataFrame([input_data])

    # Align with the model's expected features
    model_features = model.feature_names_in_

    # # debugging
    # print("Transformed Input Data Columns:", input_df.columns)
    # print("Model Features:", model_features)

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns as zero
            print("missing columns present: ", input_df[col])

    input_df = input_df[model_features]  # Ensure column order matches

    # Perform prediction
    prediction = model.predict(input_df)
    return prediction # Returns the predicted Storage ID

def evaluate_model(rf_model, X_test, y_test):
    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\n-------Classification Report-------\n", classification_report(y_test, y_pred, zero_division=0))
