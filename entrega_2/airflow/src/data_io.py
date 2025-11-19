# entrega_2/airflow/src/data_io.py

import pandas as pd

from .config import (
    RAW_DIR,
    FEATURES_PATH,
    PREDICTIONS_DIR,
    PREDICTIONS_PATH,
)


def load_raw_data():
    """
    Carga todos los datos crudos desde RAW_DIR:
    - clientes.parquet
    - productos.parquet
    - todos los transacciones*.parquet (concatenados)
    """
    clientes = pd.read_parquet(RAW_DIR / "clientes.parquet")
    productos = pd.read_parquet(RAW_DIR / "productos.parquet")

    # Cargar todas las semanas de transacciones
    tx_files = sorted(RAW_DIR.glob("transacciones*.parquet"))
    if not tx_files:
        raise FileNotFoundError(
            f"No se encontraron archivos de transacciones en {RAW_DIR}"
        )

    dfs = [pd.read_parquet(f) for f in tx_files]
    transacciones = pd.concat(dfs, ignore_index=True)

    return clientes, productos, transacciones


def save_features(df: pd.DataFrame):
    """
    Guarda el dataset de features procesadas en FEATURES_PATH.
    """
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_PATH, index=False)


def load_features() -> pd.DataFrame:
    """
    Carga el dataset de features procesadas desde FEATURES_PATH.
    """
    return pd.read_parquet(FEATURES_PATH)


def save_predictions(df: pd.DataFrame, week: int | None = None):
    """
    Guarda las predicciones en:
    - artifacts/predictions/predicciones_semana_{week}.parquet si week != None
    - artifacts/predictions/predicciones_ultima_semana.parquet si week == None
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if week is None:
        path = PREDICTIONS_PATH
    else:
        path = PREDICTIONS_DIR / f"predicciones_semana_{week}.parquet"

    df.to_parquet(path, index=False)
    print(f"[save_predictions] Predicciones guardadas en {path}")
