"""
Configuración centralizada de rutas para el pipeline SODAI.
Usa pathlib para evitar problemas de compatibilidad de rutas.
"""

from pathlib import Path

# ===============================
# Base del proyecto dentro Airflow
# ===============================
BASE_DIR = Path("/opt/airflow")

# ===============================
# Directorios de datos
# ===============================
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BATCHES_DIR = DATA_DIR / "batches"
PROCESSED_DIR = DATA_DIR / "processed"   # <<<<<< FALTABA ESTO

# ===============================
# Directorios de artefactos
# ===============================
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# Crear carpetas si no existen
for d in [RAW_DIR, BATCHES_DIR, PROCESSED_DIR, MODELS_DIR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===============================
# Archivos crudos
# ===============================
CLIENTES_PATH = RAW_DIR / "clientes.parquet"
PRODUCTOS_PATH = RAW_DIR / "productos.parquet"
# Las transacciones se cargan por batches → DATA_DIR / "batches" (*.parquet)

# ===============================
# Features procesadas
# ===============================
FEATURES_PATH = PROCESSED_DIR / "weekly_features.parquet"

# ===============================
# Modelo entrenado (default)
# ===============================
MODEL_PATH = MODELS_DIR / "xgb_best.pkl"

# ===============================
# Archivo de predicciones (fallback)
# ===============================
PREDICTIONS_PATH = PREDICTIONS_DIR / "predicciones_ultima_semana.parquet"

# ===============================
# Para compatibilidad con otros módulos
# ===============================
MODEL_DIR = MODELS_DIR              # si quieres que sea str: str(MODELS_DIR)
PREDICTIONS_OUT_DIR = PREDICTIONS_DIR
