from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
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


# Modelos para predicción de próxima semana
class CustomerProductPair(BaseModel):
    customer_id: int
    product_id: int
    prediction_proba: float
    prediction_binary: int


class WeekPredictionResponse(BaseModel):
    semana_predicha: int
    total_duplas: int
    duplas_predichas: List[CustomerProductPair]
    umbral_usado: float


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


def load_base_data():
    """
    Carga los datos base necesarios para generar combinaciones cliente-producto
    """
    try:
        # Rutas de los archivos en el backend
        current_dir = Path(__file__).resolve().parent
        
        # Intentar cargar desde múltiples ubicaciones posibles
        possible_paths = [
            current_dir / "data" / "raw",
            current_dir.parent / "data" / "raw",
            current_dir / ".." / "data" / "raw",
        ]
        
        data_dir = None
        for path in possible_paths:
            if path.exists():
                data_dir = path
                break
        
        if data_dir is None:
            # Si no encuentra los datos, usar valores por defecto
            print("ADVERTENCIA: No se encontraron datos base, usando valores simulados")
            return generate_default_combinations()
        
        # Cargar clientes y productos
        clientes = pd.read_parquet(data_dir / "clientes.parquet")
        productos = pd.read_parquet(data_dir / "productos.parquet")
        
        return clientes, productos
        
    except Exception as e:
        print(f"Error cargando datos base: {e}")
        return generate_default_combinations()


def generate_default_combinations():
    """
    Genera combinaciones por defecto cuando no hay acceso a los datos reales
    """
    # Simulamos algunos clientes y productos
    clientes = pd.DataFrame({
        'customer_id': range(1, 11),  # 10 clientes
        'customer_type': ['MINIMARKET'] * 10,
        'segment': ['MEDIUM'] * 10,
        'X': [-107.90] * 10,
        'Y': [-46.56] * 10,
        'num_deliver_per_week': [3] * 10
    })
    
    productos = pd.DataFrame({
        'product_id': range(1, 6),  # 5 productos
        'sub_category': ['GASEOSAS'] * 5,
        'package': ['BOTELLA'] * 5,
        'brand': ['Brand 1'] * 5,
        'size': [0.33] * 5
    })
    
    return clientes, productos


def create_week_combinations(clientes_df, productos_df, target_week):
    """
    Crea todas las combinaciones cliente-producto para una semana específica
    """
    # Crear todas las combinaciones posibles
    combinations = []
    
    for _, cliente in clientes_df.iterrows():
        for _, producto in productos_df.iterrows():
            combination = {
                'customer_id': cliente['customer_id'],
                'product_id': producto['product_id'],
                'semana': target_week,
                'size': producto.get('size', 0.33),
                'num_deliver_per_week': cliente.get('num_deliver_per_week', 3),
                'X': cliente.get('X', -107.90),
                'Y': cliente.get('Y', -46.56),
                'compro_semana_pasada': 0.0,  # Conservador por defecto
                'promedio_compra': 0.02,  # Mediana típica
                'customer_type': cliente.get('customer_type', 'MINIMARKET'),
                'sub_category': producto.get('sub_category', 'GASEOSAS'),
                'segment': cliente.get('segment', 'MEDIUM'),
                'package': producto.get('package', 'BOTELLA'),
                'brand': producto.get('brand', 'Brand 1'),
            }
            combinations.append(combination)
    
    return pd.DataFrame(combinations)


@app.post("/predict_next_week", response_model=WeekPredictionResponse)
def predict_next_week(threshold: float = 0.5, semana: int = 54):
    """
    Predice todas las duplas cliente-producto que comprarán en la semana especificada.
    
    Args:
        threshold: Umbral de probabilidad para considerar una compra (default 0.5)
        semana: Semana específica para hacer la predicción (default 54)
    
    Returns:
        Lista de duplas cliente-producto que se predice comprarán
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    try:
        # Usar la semana especificada por el usuario
        target_week = semana
        
        # Cargar datos base
        clientes_df, productos_df = load_base_data()
        
        # Crear todas las combinaciones para la semana objetivo
        combinations_df = create_week_combinations(clientes_df, productos_df, target_week)
        
        # Realizar predicciones para todas las combinaciones
        predictions = model.predict_proba(combinations_df)[:, 1]  # Probabilidad clase 1
        
        # Filtrar solo las duplas que superan el umbral
        high_prob_indices = predictions >= threshold
        filtered_combinations = combinations_df[high_prob_indices].copy()
        filtered_predictions = predictions[high_prob_indices]
        
        # Preparar respuesta
        duplas_predichas = []
        for i, (_, row) in enumerate(filtered_combinations.iterrows()):
            dupla = CustomerProductPair(
                customer_id=int(row['customer_id']),
                product_id=int(row['product_id']),
                prediction_proba=float(filtered_predictions[i]),
                prediction_binary=1
            )
            duplas_predichas.append(dupla)
        
        # Ordenar por probabilidad descendente
        duplas_predichas.sort(key=lambda x: x.prediction_proba, reverse=True)
        
        return WeekPredictionResponse(
            semana_predicha=target_week,
            total_duplas=len(duplas_predichas),
            duplas_predichas=duplas_predichas,
            umbral_usado=threshold
        )
        
    except Exception as e:
        print(f"Error en predicción semanal: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error procesando predicción semanal: {str(e)}"
        )
