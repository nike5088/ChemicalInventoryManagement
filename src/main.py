from fastapi import FastAPI
from src.api.endpoints import router

app = FastAPI()
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Chemical Inventory Management System"}

# debug: ensure py file runs
print("main.py is running")