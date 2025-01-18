from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from preprocess import preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data("data/chemical_inventory.csv")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    dump(model, "models/random_forest.joblib")