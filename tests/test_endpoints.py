from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Chemical Inventory Management API!" in response.json()["message"]

def test_train_endpoint():
    response = client.post("/train/")
    assert response.status_code == 200
    assert response.json() == {"status": "Model training completed successfully"}