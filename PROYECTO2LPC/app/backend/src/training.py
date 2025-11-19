# src/training.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import MODEL_PATH, ARTIFACTS_DIR
from .data_io import load_features
from .preprocessing import (
    make_preprocessor,
    TARGET_COL,
    SPLIT_COL,
)


# ============================================================
# Helpers para métricas
# ============================================================

def _compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Calcula métricas estándar a partir de probabilidades.
    Usa un threshold fijo (por defecto 0.5) para pasar a clase 0/1.
    """
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {}
    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    metrics["avg_precision"] = average_precision_score(y_true, y_proba)
    metrics["f1_at_thr"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision_pos"] = precision_score(
        y_true, y_pred, zero_division=0
    )
    metrics["recall_pos"] = recall_score(
        y_true, y_pred, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm
    metrics["threshold"] = float(threshold)
    return metrics


# ============================================================
# Entrenamiento principal
# ============================================================

def _prepare_splits(df: pd.DataFrame):
    """
    Separa el DataFrame en X_train, y_train, X_val, y_val, X_test, y_test
    usando la columna SPLIT_COL ("train", "val", "test").

    Importante: solo se saca la columna TARGET_COL; el resto
    se lo pasa al preprocesador (que luego selecciona las columnas).
    """
    # Filtramos cada split
    df_train = df[df[SPLIT_COL] == "train"].copy()
    df_val = df[df[SPLIT_COL] == "val"].copy()
    df_test = df[df[SPLIT_COL] == "test"].copy()

    # Target
    y_train = df_train[TARGET_COL].astype(int).values
    y_val = df_val[TARGET_COL].astype(int).values
    y_test = df_test[TARGET_COL].astype(int).values

    # Features: dejamos todas las columnas menos el target
    X_train = df_train.drop(columns=[TARGET_COL])
    X_val = df_val.drop(columns=[TARGET_COL])
    X_test = df_test.drop(columns=[TARGET_COL])

    return X_train, y_train, X_val, y_val, X_test, y_test


def _make_xgb_model(y_train):
    """
    Crea el modelo XGBoost con hiperparámetros razonables y
    corrige desbalance usando scale_pos_weight.
    """
    # Desbalance: proporción negativos/positivos
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    if pos > 0:
        scale_pos_weight = neg / pos
    else:
        scale_pos_weight = 1.0

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )
    return model


def train_and_evaluate() -> dict:
    """
    Función principal de entrenamiento:

    1. Carga las features desde parquet.
    2. Separa en train / val / test usando 'split'.
    3. Crea el preprocessor (ColumnTransformer) con make_preprocessor().
    4. Crea un Pipeline: preprocessor + XGBClassifier.
    5. Entrena con datos de train.
    6. Calcula métricas en val y test.
    7. Guarda el pipeline completo en MODEL_PATH.
    8. Devuelve un dict con las métricas y la ruta del modelo.
    """
    # 1) Cargar dataset de features preparado en preprocessing
    df = load_features()

    # 2) Separar splits
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = _prepare_splits(df)

    # 3) Crear preprocesador con las columnas de X_train
    preprocessor = make_preprocessor(X_train)

    # 4) Definir modelo XGBoost
    xgb_model = _make_xgb_model(y_train)

    # 5) Pipeline: preprocessor + modelo
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_model),
        ]
    )

    # 6) Entrenamiento
    clf.fit(X_train, y_train)

    # 7) Evaluación en val y test
    proba_val = clf.predict_proba(X_val)[:, 1]
    proba_test = clf.predict_proba(X_test)[:, 1]

    metrics_val = _compute_metrics(y_val, proba_val, threshold=0.5)
    metrics_test = _compute_metrics(y_test, proba_test, threshold=0.5)

    # 8) Guardar modelo entrenado (pipeline completo)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    # 9) (Opcional) Guardar métricas como JSON en artifacts
    metrics = {
        "val": metrics_val,
        "test": metrics_test,
        "model_path": str(MODEL_PATH),
    }

    metrics_path = ARTIFACTS_DIR / "metrics_train.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 10) Mostrar por pantalla (útil para logs de Airflow)
    print("Métricas (val):", metrics_val)
    print("Métricas (test):", metrics_test)
    print("Modelo guardado en:", MODEL_PATH)

    return metrics


def train_model():
    """
    Función pensada para usar directamente en el DAG de Airflow.
    Solo llama a train_and_evaluate() y devuelve las métricas.
    """
    return train_and_evaluate()
