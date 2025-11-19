# src/predict.py

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH
from .data_io import load_features, save_predictions
from .preprocessing import TARGET_COL, WEEK_COL, SPLIT_COL, ID_COLS


# ============================================================
# Helpers
# ============================================================

def _load_trained_pipeline():
    """
    Carga el pipeline completo (preprocesador + modelo) desde disco.
    Este pipeline fue guardado en training.py con joblib.dump().
    """
    clf = joblib.load(MODEL_PATH)
    return clf


def _get_next_week(weeks: np.ndarray) -> int:
    """
    Dada una lista/array de semanas presentes en los datos,
    retorna la 'próxima semana' para predecir.

    Si la última semana es 52, envuelve y pasa a 1 (para mantener ciclicidad).
    """
    max_week = int(np.max(weeks))
    next_week = max_week + 1
    if next_week > 52:
        next_week = 1
    return next_week


def _build_scoring_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la base de scoring para la 'próxima semana':

    1. Detecta la última semana disponible en el dataset.
    2. Toma todas las filas de esa semana (cliente–producto–semana).
    3. Ajusta la columna 'semana' a next_week (max_semana + 1 o 1 si >52).
    4. Elimina la columna TARGET_COL (compra_flag), si existe, porque ahora estamos prediciendo.
    """
    # 1) Última semana presente en los datos
    max_week = int(df[WEEK_COL].max())
    next_week = _get_next_week(df[WEEK_COL].values)

    # 2) Usamos la info de la última semana
    last_week_data = df[df[WEEK_COL] == max_week].copy()

    # 3) Ajustamos columna 'semana' a la próxima semana
    last_week_data[WEEK_COL] = next_week

    # 4) Eliminamos el target si está presente (no tiene sentido en predicción)
    if TARGET_COL in last_week_data.columns:
        last_week_data = last_week_data.drop(columns=[TARGET_COL])

    # Podemos quitar también la columna 'split' si queremos que no participe
    # (igual el ColumnTransformer la va a ignorar porque no la tiene mapeada).
    if SPLIT_COL in last_week_data.columns:
        last_week_data = last_week_data.drop(columns=[SPLIT_COL])

    return last_week_data, next_week


# ============================================================
# Función principal de predicción
# ============================================================

def generate_predictions(save: bool = True) -> pd.DataFrame:
    """
    Genera predicciones de compra para la 'próxima semana' y, opcionalmente,
    guarda el resultado en disco usando save_predictions().

    Flujo:
    - Carga dataset de features completo (weekly_full_final).
    - Construye base de scoring para la próxima semana.
    - Carga el pipeline entrenado (preprocesador + XGBoost).
    - Calcula proba de compra por cliente–producto.
    - Devuelve un DataFrame con: customer_id, product_id, semana, proba_compra,
      y un ranking por cliente (rank_cliente).
    """
    # 1) Cargar dataset de features
    df = load_features()

    # 2) Construir base de scoring para la próxima semana
    X_pred, next_week = _build_scoring_base(df)

    # 3) Cargar pipeline entrenado
    clf = _load_trained_pipeline()

    # 4) Obtener probabilidades de compra
    #    predict_proba devuelve 2 columnas: [proba_clase_0, proba_clase_1]
    proba_compra = clf.predict_proba(X_pred)[:, 1]

    # 5) Armar DataFrame de salida con IDs + probabilidad
    #    Nos aseguramos de tener siempre las columnas de ID
    for col in ID_COLS:
        if col not in X_pred.columns:
            raise ValueError(
                f"Columna de ID '{col}' no está presente en X_pred. "
                "Revisar pipeline de preprocesamiento."
            )

    preds = X_pred[ID_COLS].copy()
    preds[WEEK_COL] = next_week
    preds["proba_compra"] = proba_compra

    # 6) Ranking por cliente (productos más probables primero)
    preds["rank_cliente"] = (
        preds.sort_values(["customer_id", "proba_compra"], ascending=[True, False])
        .groupby("customer_id")
        .cumcount()
        + 1
    )

    # 7) Opcional: guardar a disco (parquet)
    if save:
        save_predictions(preds)

    # 8) Print útil para logs de Airflow
    print(f"Generadas {len(preds)} predicciones para semana={next_week}.")
    print("Ejemplo de salida:")
    print(preds.head())

    return preds
