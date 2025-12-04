# ============================================
# src/data_io.py — versión corregida para batches
# ============================================

import pandas as pd
from pathlib import Path

from .config import (
    RAW_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    PREDICTIONS_DIR,
    DATA_DIR,
    BATCHES_DIR,   # <<<<<< AÑADIR
)


# ============================================================
# CARGA DE DATOS RAW + BATCHES
# ============================================================
def load_raw_data():
    df_clientes = pd.read_parquet(RAW_DIR / "clientes.parquet")
    df_productos = pd.read_parquet(RAW_DIR / "productos.parquet")

    # cargar todos los batches
    batch_files = sorted(BATCHES_DIR.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in batch_files]
    df_transacciones = pd.concat(dfs, ignore_index=True)

    return df_clientes, df_productos, df_transacciones



# ============================================================
# GUARDAR FEATURES Y PREDICCIONES
# ============================================================

def save_features(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "weekly_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[save_features] Features guardadas en {out_path}")
    return out_path


def load_features():
    path = PROCESSED_DIR / "weekly_features.parquet"
    return pd.read_parquet(path)


def save_predictions(df, week: str):
    """
    Guarda predicciones estructuradas de cada batch en parquet.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"predicciones_semana_{week}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[save_predictions] Predicciones guardadas en {out_path}")
    return out_path
