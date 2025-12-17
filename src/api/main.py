import mlflow
import numpy as np
from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API")

MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    X = np.array(request.features).reshape(1, -1)
    proba = model.predict(X)[0]
    return {"risk_probability": float(proba)}
