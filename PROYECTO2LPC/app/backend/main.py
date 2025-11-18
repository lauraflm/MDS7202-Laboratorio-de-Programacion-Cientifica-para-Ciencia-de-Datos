# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from model_utils import load_model

app = FastAPI(
    title="SodAI Drinks Backend",
    description="API para servir predicciones del modelo de SodAI Drinks.",
    version="1.0.0",
)

model = load_model()


class DrinkFeatures(BaseModel):
    sugar_grams: float
    acidity: float
    carbonation: float
    caffeine_mg: float


class PredictionResponse(BaseModel):
    quality_label: Literal["low", "medium", "high"]
    quality_score: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: DrinkFeatures):
    X = [[
        features.sugar_grams,
        features.acidity,
        features.carbonation,
        features.caffeine_mg,
    ]]
    score = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(X)[0])

    if score < 0.33:
        label = "low"
    elif score < 0.66:
        label = "medium"
    else:
        label = "high"

    return {
        "quality_label": label,
        "quality_score": score,
    }
