import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Handle missing values, encode categorical variables, normalize features, etc.
    df = df.fillna("Unknown")
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)