from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel, Field
from src.ml.predict import predict
from src.ml.train import train_model
from datetime import datetime
from typing import Optional

router = APIRouter()

class ChemicalData(BaseModel):
    ChemicalName: str = Field(..., example="Acetone", description="The scientific name of chemical")
    CASNumber: str = Field(..., example="123-45-6", description="The CAS number of chemical")
    Hazard: str = Field(..., example="Corosive", description="The hazardous classification of chemical")
    StorageTemp: str = Field(...,example="Refrigerator", description="The recommended storage temperature")
    Container: str = Field(...,example="Glass",description="bottle type")
    Amount: str = Field(..., example="250 g", description="The amount of chemical in a container")
    Owner: str = Field(..., example="PJ", description="Owner Initials")
    Manufacturer: str = Field(..., example="Sigma-Aldrich", description="The manufacturer of chemical")
    ReceivedDate: Optional[datetime] = Field(..., example="1/1/2020")


@router.get("/")
def home():
    return {"message": "Welcome to the Chemical Inventory Management API!"}

@router.post("/classify/", tags=["Prediction"])
def predict_chemical(data: ChemicalData):
    try:
        input_data = data.dict()
        prediction = predict(input_data)
        return {"classification": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/train/")
def train():
    try:
        train_model()
        return {"status": "Model training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))