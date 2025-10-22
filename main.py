from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import uvicorn
import os

DATA_PATH = os.getenv("DATA_PATH", "/app/data/water_potability.csv")


app = FastAPI(
    title="API de Predicción de Potabilidad del Agua",
    description="Sistema de predicción para determinar si el agua es potable o no potable",
    version="1.0.0"
)

class WaterQualityData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float
    
    class Config:
        # Ejemplo de datos
        model_config = {
        "json_schema_extra": {
            "example": {
                "ph": 10.316400384553162,
                "Hardness": 217.2668424334475,
                "Solids": 10676.508475429378,
                "Chloramines": 3.445514571005745,
                "Sulfate": 397.7549459751925,
                "Conductivity": 492.20647361771086,
                "Organic_carbon": 12.812732207582542,
                "Trihalomethanes": 72.28192021570328,
                "Turbidity": 3.4073494284238364
            }
        }
    }


class PredictionResponse(BaseModel):
    potabilidad: int

# Cargar el modelo entrenado al iniciar la aplicación
def load_model():
    """Cargar el mejor modelo entrenado"""
    try:
        model_path = "models/best_xgboost_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Modelo cargado exitosamente desde {model_path}")
            return model
        else:
            print(f"No se encontró el modelo en {model_path}")
            return None
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None

model = load_model()


def load_imputer():
    """Cargar y preparar el imputer basado en los datos de entrenamiento"""
    try:
        data = pd.read_csv(DATA_PATH)
        X = data.drop('Potability', axis=1)
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X)
        return imputer
    except Exception as e:
        print(f"Error preparando el imputer: {e}")
        return None


imputer = load_imputer()

@app.get("/")
def home():
    """
    Endpoint principal que describe el modelo y sistema
    """
    return {
        "mensaje": "API de Predicción de Potabilidad del Agua",
        "descripción": {
            "problema": "Determinar si el agua es potable o no potable basándose en mediciones químicas",
            "modelo": "XGBoost optimizado con Optuna",
            "entrada": {
                "descripción": "9 parámetros químicos del agua",
                "parámetros": [
                    "pH value - Nivel de acidez/alcalinidad",
                    "Hardness - Dureza del agua",
                    "Solids - Sólidos totales disueltos (TDS)",
                    "Chloramines - Nivel de cloraminas",
                    "Sulfate - Concentración de sulfatos",
                    "Conductivity - Conductividad eléctrica",
                    "Organic_carbon - Carbono orgánico",
                    "Trihalomethanes - Trihalometanos",
                    "Turbidity - Turbidez del agua"
                ]
            },
            "salida": {
                "potabilidad": "0 = No potable, 1 = Potable"
            }
        },
        "uso": {
            "predicción": "POST /potabilidad/ con los 9 parámetros químicos",
            "documentación": "GET /docs para interfaz interactiva"
        },
        "estado_modelo": "Cargado" if model is not None else "No disponible"
    }

@app.post("/potabilidad/", response_model=PredictionResponse)
def predict_potability(data: WaterQualityData):
    """
    Predice si el agua es potable o no basándose en los parámetros químicos
    
    Args:
        data: Datos de calidad del agua con 9 parámetros químicos
        
    Returns:
        PredictionResponse: Resultado de la predicción (0 = No potable, 1 = Potable)
    """
    if model is None:
        return {"error": "Modelo no disponible. Asegúrate de haber entrenado el modelo primero."}
    
    if imputer is None:
        return {"error": "Imputer no disponible. Error en la preparación de datos."}
    
    try:
        # Convertir datos de entrada a formato pandas DataFrame
        input_data = pd.DataFrame({
            'ph': [data.ph],
            'Hardness': [data.Hardness],
            'Solids': [data.Solids],
            'Chloramines': [data.Chloramines],
            'Sulfate': [data.Sulfate],
            'Conductivity': [data.Conductivity],
            'Organic_carbon': [data.Organic_carbon],
            'Trihalomethanes': [data.Trihalomethanes],
            'Turbidity': [data.Turbidity]
        })
        
        # Aplicar imputación de valores faltantes (misma estrategia que en entrenamiento)
        input_imputed = imputer.transform(input_data)
        
        # Realizar predicción
        prediction = model.predict(input_imputed)[0]
        
        # Convertir a int estándar de Python (no numpy.int64)
        prediction_int = int(prediction)
        
        return PredictionResponse(potabilidad=prediction_int)
        
    except Exception as e:
        return {"error": f"Error en la predicción: {str(e)}"}

@app.get("/health")
def health_check():
    """
    Endpoint de verificación de estado de la API
    """
    return {
        "status": "healthy",
        "modelo_cargado": model is not None,
        "imputer_cargado": imputer is not None
    }

@app.get("/info")
def model_info():
    """
    Información detallada sobre el modelo cargado
    """
    if model is None:
        return {"error": "Modelo no disponible"}
    
    try:
        # Información básica del modelo
        model_info = {
            "tipo_modelo": str(type(model).__name__),
            "características_entrada": 9,
            "nombres_características": [
                "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
            ]
        }
        
        # Si es un modelo XGBoost, agregar información específica
        if hasattr(model, 'get_params'):
            params = model.get_params()
            model_info["parámetros"] = params
            
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_.tolist()
            feature_names = model_info["nombres_características"]
            feature_importance = {name: importance for name, importance in zip(feature_names, importances)}
            model_info["importancia_características"] = feature_importance
        
        return model_info
        
    except Exception as e:
        return {"error": f"Error obteniendo información del modelo: {str(e)}"}

# Función para ejecutar el servidor
def run_server():
    """Ejecutar el servidor FastAPI"""
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    print("API disponible en: http://localhost:8000")
    run_server()