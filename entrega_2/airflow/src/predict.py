# ============================================
# src/predict.py — versión final CodaLab + batches
# ============================================

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from .config import MODEL_PATH, PREDICTIONS_DIR
from .data_io import load_features, save_predictions
from .preprocessing import TARGET_COL, WEEK_COL, SPLIT_COL


# ============================================================
# Semana siguiente según datos reales
# ============================================================

def _get_next_week(weeks):
    """Retorna next_week = max_week + 1, con wrap 52 → 1."""
    max_week = int(np.max(weeks))
    next_week = max_week + 1
    return 1 if next_week > 52 else next_week


def _make_scoring_week(df):
    """
    Construye la base de scoring REAL:
    - Toma la última semana histórica (incluye batches)
    - La copia como semana siguiente
    - Elimina columnas objetivo (target/split)
    """
    last_real_week = int(df[WEEK_COL].max())
    next_week = _get_next_week(df[WEEK_COL].values)

    df_last = df[df[WEEK_COL] == last_real_week].copy()
    df_last[WEEK_COL] = next_week

    # Eliminar columnas que no sirven para scoring
    for col in [TARGET_COL, SPLIT_COL]:
        if col in df_last.columns:
            df_last = df_last.drop(columns=[col])

    return df_last, next_week


# ============================================================
# Guardar CSV exacto para CodaLab
# ============================================================

def _save_codalab_csv(df, batch_id):
    """Guarda CSV sin header (formato obligatorio CodaLab)."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = PREDICTIONS_DIR / f"predictions_{batch_id}.csv"
    df.to_csv(file_path, index=False, header=False)
    print(f"[CodaLab] Archivo guardado en: {file_path}")
    return file_path


# ============================================================
# GENERATE PREDICTIONS — versión final
# ============================================================

def generate_predictions(batch_id: str):
    """
    Flujo real de predicciones SodAI+CodaLab:
    - Carga todas las features ya construidas (incluye batches)
    - Identifica la última semana real
    - Construye semana futura para scoring
    - Aplica modelo productivo (pipeline completo)
    - Guarda CSV EXACTO para CodaLab
    """

    print(f"[PREDICT] generate_predictions llamado con batch_id={batch_id}")

    # 1) Cargar features completas (ya incluyen batches)
    df = load_features()

    # 2) Crear semana futura de scoring
    df_scoring, next_week = _make_scoring_week(df)

    print(f"[PREDICT] Base de scoring: {len(df_scoring)} filas — semana siguiente = {next_week}")

    # 3) Cargar pipeline completo entrenado
    print(f"[PREDICT] Cargando modelo desde: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # 4) Predecir probabilidades
    X = df_scoring.drop(columns=[c for c in [TARGET_COL] if c in df_scoring.columns])
    proba = model.predict_proba(X)[:, 1]

    df_scoring["score"] = proba

    # 5) Ordenar descendente (más probable primero)
    df_scoring = df_scoring.sort_values("score", ascending=False)

    # 6) Exportar ONLY customer_id, product_id
    df_codalab = df_scoring[["customer_id", "product_id"]].copy()

    # 7) Guardar CSV EXACTO para CodaLab
    csv_path = _save_codalab_csv(df_codalab, batch_id)

    # 8) Guardar versión interna en parquet
    save_predictions(df_codalab, week=batch_id)

    print("[PREDICT] Predicciones finalizadas y guardadas.")

