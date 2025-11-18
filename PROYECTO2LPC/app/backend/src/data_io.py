# entrega_2/airflow/src/data_io.py
import pandas as pd
from .config import (
    CLIENTES_PATH, PRODUCTOS_PATH, TRANSACCIONES_PATH,
    FEATURES_PATH, PREDICTIONS_PATH
)

def load_raw_data():
    df_clientes = pd.read_parquet(CLIENTES_PATH)
    df_productos = pd.read_parquet(PRODUCTOS_PATH)
    df_transacciones = pd.read_parquet(TRANSACCIONES_PATH)
    return df_clientes, df_productos, df_transacciones

def save_features(df):
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)

def load_features():
    return pd.read_parquet(FEATURES_PATH)

def save_predictions(df):
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PREDICTIONS_PATH, index=False)
