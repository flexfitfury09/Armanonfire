from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df).tolist()
    return {"prediction": pred}
