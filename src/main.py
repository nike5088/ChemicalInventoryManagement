from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Chemical Inventory Management System"}

# debug: ensure py file runs
print("main.py running")