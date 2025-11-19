# entrega_2/airflow/src/config.py
from pathlib import Path

# Base del proyecto dentro del contenedor de Airflow
BASE_DIR = Path("/opt/airflow")

# Directorios principales
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Subdirectorios
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"

# Archivos de entrada (datos crudos)
CLIENTES_PATH = RAW_DIR / "clientes.parquet"
PRODUCTOS_PATH = RAW_DIR / "productos.parquet"
# No usamos TRANSACCIONES_PATH fijo porque cargamos TODOS los transacciones*.parquet
# TRANSACCIONES_PATH = RAW_DIR / "transacciones.parquet"

# Archivos generados en pipeline
FEATURES_PATH = PROCESSED_DIR / "weekly_features.parquet"

# Modelo entrenado
MODEL_PATH = MODELS_DIR / "xgb_best.pkl"

# Archivo default (por si quieres tener uno "último disponible")
# Pero ahora las predicciones se guardarán como:
# artifacts/predictions/predicciones_semana_{semana}.parquet
PREDICTIONS_PATH = PREDICTIONS_DIR / "predicciones_ultima_semana.parquet"
