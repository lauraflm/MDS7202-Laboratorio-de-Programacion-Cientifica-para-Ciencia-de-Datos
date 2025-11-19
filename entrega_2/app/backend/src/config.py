# entrega_2/airflow/src/config.py
from pathlib import Path

BASE_DIR = Path("/opt/airflow") 
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# nombres de archivos
CLIENTES_PATH = RAW_DIR / "clientes.parquet"
PRODUCTOS_PATH = RAW_DIR / "productos.parquet"
TRANSACCIONES_PATH = RAW_DIR / "transacciones.parquet"

FEATURES_PATH = PROCESSED_DIR / "weekly_features.parquet"
MODEL_PATH = MODELS_DIR / "xgb_best.pkl"
PREDICTIONS_PATH = PREDICTIONS_DIR / "predicciones_proxima_semana.parquet"
