from fastapi import APIRouter
from ml.predict import predict

router = APIRouter()

@router.post("/classify/")
def classify_chemical(data: dict):
    prediction = predict(data)
    return {"classification": prediction}