from joblib import load

def predict(input_data):
    # load the model
    model = load("models/random_forest.joblib")

    # perform prediction
    prediction = model.predict([input_data])
    return prediction