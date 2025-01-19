from joblib import load

def predict(input_data):
    model = load("models/random_forest.joblib")
    return model.predict([input_data])