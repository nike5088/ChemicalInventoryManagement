from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import pandas as pd

def train_model():
    #load data
    df = pd.read_csv("data/chemical_inventory.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    #train the model
    model = RandomForestClassifier()
    model.fit(X,y)

    #save the trained model
    dump(model, "models/random_forest.joblib")