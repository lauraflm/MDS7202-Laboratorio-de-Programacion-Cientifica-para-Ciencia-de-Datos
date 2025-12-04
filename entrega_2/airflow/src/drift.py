# src/drift.py

import numpy as np
import pandas as pd

from .data_io import load_features
from .preprocessing import WEEK_COL, SPLIT_COL, TARGET_COL


# ========================= Helpers internos =========================

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


# ========================= Calidad de datos =========================

def check_data_quality(
    batch_id: str,
    max_null_ratio: float = 0.2,
    min_rows: int = 1000,
):
    """
    Revisa la calidad de datos del batch "reciente".

    NOTA: por simplicidad no mapeamos batch_id -> semana,
    sino que tomamos la última semana disponible en las features
    como "batch actual". Si quieres algo más preciso, puedes
    adaptar esto a tu lógica de batch.

    Parámetros:
        batch_id (str): ID de batch (no se usa directamente aquí).
        max_null_ratio (float): máximo porcentaje promedio de nulos permitido.
        min_rows (int): mínimo número de filas esperado.

    Retorna:
        dict con métricas básicas de calidad.
    """
    df = load_features()

    if df.empty:
        print("[QUALITY] WARNING: DataFrame de features vacío.")
        metrics = {
            "row_count": 0,
            "mean_null_ratio": 1.0,
            "pass_quality": False,
        }
        _log_quality_metrics(metrics)
        return metrics

    # Tomamos como "batch actual" la última semana
    max_week = df[WEEK_COL].max()
    df_batch = df[df[WEEK_COL] == max_week].copy()

    row_count = len(df_batch)
    # porcentaje promedio de nulos (promediando por columna)
    mean_null_ratio = float(df_batch.isna().mean().mean())

    pass_quality = (row_count >= min_rows) and (mean_null_ratio <= max_null_ratio)

    print("[QUALITY] Evaluando calidad de datos del batch actual.")
    print(f"[QUALITY] Semana considerada: {max_week}")
    print(f"[QUALITY] Filas en el batch: {row_count}")
    print(f"[QUALITY] Porcentaje promedio de nulos: {mean_null_ratio:.4f}")
    print(f"[QUALITY] ¿Pasa umbrales? {pass_quality} "
          f"(min_rows={min_rows}, max_null_ratio={max_null_ratio})")

    metrics = {
        "row_count": row_count,
        "mean_null_ratio": mean_null_ratio,
        "pass_quality": pass_quality,
    }

    _log_quality_metrics(metrics)
    return metrics


def _log_quality_metrics(metrics: dict):
    """
    Loguea las métricas de calidad en MLflow si está disponible.
    Si no, simplemente hace print.
    """
    try:
        import mlflow

        mlflow.log_metric("quality_row_count", metrics["row_count"])
        mlflow.log_metric("quality_mean_null_ratio", metrics["mean_null_ratio"])
        mlflow.log_metric(
            "quality_pass",
            1.0 if metrics["pass_quality"] else 0.0,
        )
    except Exception as e:
        # Si no hay MLflow o no está configurado, no rompemos el pipeline.
        print(f"[QUALITY] No se pudieron loguear métricas en MLflow: {e}")


# ========================= Drift de datos =========================

def detect_drift(
    weeks_recent: int = 4,
    z_threshold: float = 0.5,
    min_columns_exceeding: int = 2,
    max_rows: int = 200_000,   # <<< NUEVO: límite para evitar OOM
) -> bool:
    """
    Detecta drift comparando mean/std de columnas numéricas entre train y semanas recientes.
    Con SAMPLE para evitar OUT-OF-MEMORY en Airflow.
    """

    df = load_features()

    # === SAMPLE EARLY PARA EVITAR MATAR DOCKER ===
    if len(df) > max_rows:
        print(f"[DRIFT] DataFrame tiene {len(df)} filas → se reduce a {max_rows}.")
        df = df.sample(max_rows, random_state=42)

    # --- separa train y datos recientes ---
    df_train = df[df[SPLIT_COL] == "train"].copy()

    max_week = int(df[WEEK_COL].max())
    cutoff_week = max_week - weeks_recent + 1
    df_recent = df[df[WEEK_COL] >= cutoff_week].copy()

    # === SAMPLE también en train y recent ===
    if len(df_train) > max_rows:
        df_train = df_train.sample(max_rows, random_state=42)

    if len(df_recent) > max_rows:
        df_recent = df_recent.sample(max_rows, random_state=42)

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

    print("[DRIFT] Columnas que superan el threshold:", cols_exceeding)

    has_drift = len(cols_exceeding) >= min_columns_exceeding
    _log_drift_metrics(has_drift, len(cols_exceeding))

    return has_drift



def _log_drift_metrics(has_drift: bool, n_cols_exceeding: int):
    """
    Loguea métricas de drift en MLflow si está disponible.
    """
    try:
        import mlflow

        mlflow.log_metric("drift_detected", 1.0 if has_drift else 0.0)
        mlflow.log_metric("drift_n_columns_exceeding", n_cols_exceeding)
    except Exception as e:
        print(f"[DRIFT] No se pudieron loguear métricas en MLflow: {e}")


def check_data_drift(batch_id: str):
    """
    Wrapper que usa detect_drift para el DAG de Airflow.

    batch_id se recibe desde el DAG, pero aquí usamos la lógica
    de "últimas semanas" ya definida en detect_drift.
    """
    print(f"[DRIFT] check_data_drift llamado para batch_id={batch_id}")
    has_drift = detect_drift()
    return {"has_drift": has_drift}


# ========================= Decisión de reentrenar =========================

def decide_retraining(
    batch_id: str,
    force_train_if_drift: bool = True,
) -> str:
    """
    Decide si reentrenar o no el modelo.

    Estrategia simple:
      - Si detect_drift() es True y force_train_if_drift = True,
        se devuelve "train_model".
      - En caso contrario, "skip_train".

    Airflow BranchPythonOperator espera que esto devuelva
    el task_id al que debe ir:
      - "train_model"
      - "skip_train"
    """
    print(f"[BRANCH] decide_retraining llamado para batch_id={batch_id}")

    has_drift = detect_drift()

    if force_train_if_drift and has_drift:
        decision = "train_model"
    else:
        decision = "skip_train"

    print(f"[BRANCH] has_drift={has_drift} -> decisión={decision}")

    # Opcional: loguear la decisión en MLflow
    try:
        import mlflow
        mlflow.log_metric("retrain_decision_train_model",
                          1.0 if decision == "train_model" else 0.0)
    except Exception as e:
        print(f"[BRANCH] No se pudo loguear decisión en MLflow: {e}")

    return decision
