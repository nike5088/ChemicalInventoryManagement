from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Chemical Inventory Management System API"}

def test_train():
    response = client.get("/train")
    assert response.status_code == 200
    assert response.json() == {"message": "Training complete"}