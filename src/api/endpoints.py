from fastapi import APIRouter
from src.ml.predict import predict
from src.ml.train import train_model

router = APIRouter()

@router.get("/")
def home():
    return {"message": "Welcome to the Chemical Inventory Management API!"}

@router.post("/classify/")
def classify_chemical(data: dict):
    prediction = predict(data)
    return {"classification": prediction}

@router.post("/train/")
def train():
    train_model()
    return {"status": "Model training completed successfully"}