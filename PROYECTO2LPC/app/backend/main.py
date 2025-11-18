from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import pandas as pd
import numpy as np
import traceback
from model_utils import load_model


app = FastAPI(
    title="SodAI Drinks Backend",
    description="API para servir predicciones del modelo entrenado de SodAI Drinks.",
    version="1.0.0",
)

# Cargar el modelo
model = load_model()
print("Modelo cargado")


# Entrada: features individuales para crear una predicción
class PredictionRequest(BaseModel):
    customer_id: int
    product_id: int
    semana: int = 1  # Semana para la predicción (por defecto 1)
    
    # Features numéricos (con valores por defecto basados en el dataset real)
    size: Optional[float] = 0.33  # Mediana del dataset
    num_deliver_per_week: Optional[float] = 3.0  # Mediana del dataset
    X: Optional[float] = -107.90  # Mediana del dataset (coordenada)
    Y: Optional[float] = -46.56   # Mediana del dataset (coordenada)
    compro_semana_pasada: Optional[float] = 0.0  # Valor más conservador
    promedio_compra: Optional[float] = 0.02  # Mediana del dataset
    
    # Features categóricos (valores exactos del dataset)
    customer_type: Optional[str] = "MINIMARKET"  # Valor común
    sub_category: Optional[str] = "GASEOSAS"     # Una de las dos opciones
    segment: Optional[str] = "MEDIUM"            # Valor intermedio
    package: Optional[str] = "BOTELLA"          # Tipo más común
    brand: Optional[str] = "Brand 1"            # Primera marca disponible


# Entrada alternativa: diccionario de features (compatibilidad)
class SimplePredictionRequest(BaseModel):
    features: Dict[str, float]


# Salida: predicción con probabilidad
class PredictionResponse(BaseModel):
    customer_id: int
    product_id: int
    prediction_proba: float
    prediction_binary: int  # 0 o 1 basado en umbral 0.5


@app.get("/")
def root():
    return {"message": "SodAI Drinks API - Ready to predict!", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """
    Endpoint principal para predicciones individuales.
    Recibe features de un cliente-producto específico.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Manejar valores None usando los valores por defecto del modelo
        size = req.size if req.size is not None else 0.33
        num_deliver = req.num_deliver_per_week if req.num_deliver_per_week is not None else 3.0
        x_coord = req.X if req.X is not None else -107.90
        y_coord = req.Y if req.Y is not None else -46.56
        
        # Crear DataFrame con la estructura esperada por el modelo
        input_data = pd.DataFrame([{
            'customer_id': req.customer_id,
            'product_id': req.product_id,
            'semana': req.semana,
            'size': size,
            'num_deliver_per_week': num_deliver,
            'X': x_coord,
            'Y': y_coord,
            'compro_semana_pasada': req.compro_semana_pasada,
            'promedio_compra': req.promedio_compra,
            'customer_type': req.customer_type,
            'sub_category': req.sub_category,
            'segment': req.segment,
            'package': req.package,
            'brand': req.brand,
        }])
        
        print(f"DEBUG: Datos de entrada: {input_data.to_dict('records')[0]}")  # Para debug
        
        # Realizar predicción (probabilidad de compra)
        pred_proba = model.predict_proba(input_data)[0, 1]  # Probabilidad clase 1
        pred_binary = 1 if pred_proba >= 0.5 else 0
        
        return PredictionResponse(
            customer_id=req.customer_id,
            product_id=req.product_id,
            prediction_proba=float(pred_proba),
            prediction_binary=pred_binary
        )
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error procesando predicción: {str(e)}"
        )


@app.post("/predict_simple", response_model=Dict)
def predict_simple(req: SimplePredictionRequest):
    """
    Endpoint alternativo para compatibilidad con frontend existente.
    Recibe un diccionario de features.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Convertir dict a DataFrame
        df = pd.DataFrame([req.features])
        
        # Asegurar que customer_id y product_id estén presentes
        if 'customer_id' not in df.columns:
            df['customer_id'] = 1  # Valor por defecto
        if 'product_id' not in df.columns:
            df['product_id'] = 1  # Valor por defecto
        if 'semana' not in df.columns:
            df['semana'] = 1  # Valor por defecto
        
        # Realizar predicción
        pred_proba = model.predict_proba(df)[0, 1]
        
        return {
            "prediction": float(pred_proba),
            "prediction_binary": 1 if pred_proba >= 0.5 else 0,
            "message": "Predicción exitosa"
        }
        
    except Exception as e:
        print(f"Error en predicción simple: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error procesando predicción: {str(e)}"
        )
