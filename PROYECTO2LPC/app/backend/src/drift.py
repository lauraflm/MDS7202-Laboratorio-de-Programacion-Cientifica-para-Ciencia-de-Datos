# src/drift.py

import numpy as np
import pandas as pd

from .data_io import load_features
from .preprocessing import WEEK_COL, SPLIT_COL, TARGET_COL


def _select_numeric_columns(df: pd.DataFrame):
    """
    Selecciona columnas numéricas relevantes para monitorear drift,
    excluyendo el target y la semana (que ya sabemos que cambia).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in [TARGET_COL, WEEK_COL]:
        if col in num_cols:
            num_cols.remove(col)

    return num_cols


def detect_drift(
    weeks_recent: int = 4,
    z_threshold: float = 0.5,
    min_columns_exceeding: int = 2,
) -> bool:
    """
    Detecta drift en los datos de forma simple:

    - Usa como "baseline" el split de entrenamiento (split == "train").
    - Usa como "datos recientes" las últimas `weeks_recent` semanas.
    - Para cada columna numérica, compara:
        z = |mean_recent - mean_train| / std_train
      Si z > z_threshold, se considera que hay cambio en esa columna.
    - Si al menos `min_columns_exceeding` columnas superan el umbral,
      devuelve True (hay drift), si no devuelve False.

    Retorna:
        bool: True si se detecta drift, False en caso contrario.
    """
    df = load_features()

    # --- separa train y datos recientes ---
    df_train = df[df[SPLIT_COL] == "train"].copy()

    max_week = int(df[WEEK_COL].max())
    cutoff_week = max_week - weeks_recent + 1
    df_recent = df[df[WEEK_COL] >= cutoff_week].copy()

    if df_train.empty or df_recent.empty:
        print("WARNING: df_train o df_recent está vacío; no se evalúa drift.")
        return False

    num_cols = _select_numeric_columns(df)

    if not num_cols:
        print("WARNING: no hay columnas numéricas para monitorear drift.")
        return False

    print(f"[DRIFT] Comparando train vs últimas {weeks_recent} semanas.")
    print(f"[DRIFT] Columnas numéricas monitoreadas: {num_cols}")

    cols_exceeding = []
    drift_report = {}

    for col in num_cols:
        train_col = df_train[col].dropna()
        recent_col = df_recent[col].dropna()

        if train_col.empty or recent_col.empty:
            continue

        mu_train = train_col.mean()
        std_train = train_col.std(ddof=1)
        mu_recent = recent_col.mean()

        if std_train == 0 or np.isnan(std_train):
            # si no hay variación en train, skip
            continue

        z = abs(mu_recent - mu_train) / std_train

        drift_report[col] = {
            "mu_train": float(mu_train),
            "mu_recent": float(mu_recent),
            "std_train": float(std_train),
            "z_score": float(z),
        }

        if z > z_threshold:
            cols_exceeding.append(col)

    print("[DRIFT] Reporte por columna:")
    for col, info in drift_report.items():
        print(
            f"  {col}: mu_train={info['mu_train']:.4f}, "
            f"mu_recent={info['mu_recent']:.4f}, "
            f"z={info['z_score']:.3f}"
        )

    print(
        f"[DRIFT] Columnas que superan z_threshold={z_threshold}: "
        f"{cols_exceeding}"
    )

    has_drift = len(cols_exceeding) >= min_columns_exceeding

    if has_drift:
        print(
            f"[DRIFT] Drift DETECTADO "
            f"(cols_exceeding={len(cols_exceeding)} >= {min_columns_exceeding})."
        )
    else:
        print(
            f"[DRIFT] Drift NO detectado "
            f"(cols_exceeding={len(cols_exceeding)} < {min_columns_exceeding})."
        )

    return has_drift
