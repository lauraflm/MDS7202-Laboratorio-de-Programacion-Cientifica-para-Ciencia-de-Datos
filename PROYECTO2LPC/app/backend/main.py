from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd

from model_utils import load_model


app = FastAPI(
    title="SodAI Drinks Backend",
    description="API para servir predicciones del modelo entrenado de SodAI Drinks.",
    version="1.0.0",
)

# Cargar el modelo
model = load_model()
print("Modelo cargado")


# Entrada: diccionario con los features del modelo
class PredictionRequest(BaseModel):
    features: Dict[str, float]


# Salida ->> pred
class PredictionResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    df = pd.DataFrame([req.features])
    pred = model.predict(df)[0]

    return PredictionResponse(prediction=float(pred))
