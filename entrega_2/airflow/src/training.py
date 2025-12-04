# src/training.py

import json
import os
from pathlib import Path
import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn  # para modelos tipo scikit-learn / XGBoost

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

# SHAP opcional
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Se usarán solo feature importances básicas.")

from .config import MODEL_PATH, ARTIFACTS_DIR
from .data_io import load_features
from .preprocessing import make_preprocessor, TARGET_COL, SPLIT_COL

# ============================================================
# Config MLflow
# ============================================================

MLRUNS_PATH = ARTIFACTS_DIR / "mlruns"
mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
mlflow.set_experiment("sodai-entrega3")


# ============================================================
# Helpers para interpretabilidad
# ============================================================

def _generate_interpretability_plots(model, X_test, y_test, plots_dir):
    """
    Genera algunos gráficos de interpretabilidad del modelo.
    """
    interpretability_results = {}

    try:
        os.makedirs(plots_dir, exist_ok=True)

        # --- 1. Feature importance básica ---
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
            feature_names = (
                X_test.columns
                if hasattr(X_test, "columns")
                else [f"feature_{i}" for i in range(X_test.shape[1])]
            )

            importance_df = (
                pd.DataFrame(
                    {"feature": feature_names, "importance": feature_importance}
                )
                .sort_values("importance", ascending=False)
            )

            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)

            bars = plt.barh(range(len(top_features)), top_features["importance"])
            plt.yticks(range(len(top_features)), top_features["feature"])
            plt.xlabel("Feature Importance")
            plt.title("Top 15 Feature Importance - XGBoost Model")
            plt.gca().invert_yaxis()

            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}",
                    ha="left",
                    va="center",
                    fontsize=8,
                )

            plt.tight_layout()
            feature_importance_path = os.path.join(
                plots_dir, "feature_importance.png"
            )
            plt.savefig(feature_importance_path, dpi=300, bbox_inches="tight")
            plt.close()

            interpretability_results["feature_importance_plot"] = feature_importance_path
            interpretability_results["top_features"] = top_features.to_dict("records")[
                :10
            ]

        # --- 2. Confusion matrix ---
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()

        class_names = ["No Compra", "Compra"]
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        confusion_matrix_path = os.path.join(plots_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretability_results["confusion_matrix_plot"] = confusion_matrix_path
        interpretability_results["confusion_matrix"] = cm.tolist()

    except Exception as e:
        interpretability_results["plots_generation"] = f"failed: {str(e)}"
        print(f"Error generating interpretability plots: {e}")

    return interpretability_results


def _save_comprehensive_tracking(
    metrics,
    hyperparams,
    interpretability_results,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Guarda información del experimento en un JSON.
    """
    experiment_info = {
        "experiment_id": f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": "XGBoost",
        "model_version": "v1.0",
        "data_version": "weekly_features_v1",
    }

    data_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1]
        if hasattr(X_train, "shape")
        else len(X_train[0]),
        "class_distribution_train": {
            "class_0": int((y_train == 0).sum()),
            "class_1": int((y_train == 1).sum()),
        },
        "class_distribution_test": {
            "class_0": int((y_test == 0).sum()),
            "class_1": int((y_test == 1).sum()),
        },
    }

    comprehensive_tracking = {
        "experiment_info": experiment_info,
        "hyperparameters": hyperparams,
        "metrics": metrics,
        "data_info": data_info,
        "interpretability": interpretability_results,
    }

    tracking_path = ARTIFACTS_DIR / "experiment_tracking.json"
    tracking_path.parent.mkdir(parents=True, exist_ok=True)

    with open(tracking_path, "w") as f:
        json.dump(comprehensive_tracking, f, indent=2, ensure_ascii=False)

    print(f"Comprehensive experiment tracking saved to: {tracking_path}")
    return str(tracking_path)


def _compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Calcula métricas estándar a partir de probabilidades.
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
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["threshold"] = float(threshold)
    return metrics


# ============================================================
# Entrenamiento principal
# ============================================================

def _prepare_splits(df: pd.DataFrame):
    """
    Separa en train / val / test usando SPLIT_COL.
    """
    df_train = df[df[SPLIT_COL] == "train"].copy()
    df_val = df[df[SPLIT_COL] == "val"].copy()
    df_test = df[df[SPLIT_COL] == "test"].copy()

    y_train = df_train[TARGET_COL].astype(int).values
    y_val = df_val[TARGET_COL].astype(int).values
    y_test = df_test[TARGET_COL].astype(int).values

    X_train = df_train.drop(columns=[TARGET_COL])
    X_val = df_val.drop(columns=[TARGET_COL])
    X_test = df_test.drop(columns=[TARGET_COL])

    return X_train, y_train, X_val, y_val, X_test, y_test


def _make_xgb_model(y_train):
    """
    Crea el modelo XGBoost con scale_pos_weight para desbalance.
    """
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2.0,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        reg_lambda=2.0,
        reg_alpha=0.5,
    )
    return model


def train_and_evaluate(batch_id: str | None = None) -> dict:
    """
    Entrena el pipeline (preprocesador + XGBoost) y lo guarda en MODEL_PATH.
    """
    df = load_features()

    # df es la tabla completa con split
    df_train = df[df[SPLIT_COL] == "train"].copy()

    df_pos = df_train[df_train[TARGET_COL] == 1]
    df_neg = df_train[df_train[TARGET_COL] == 0]

    neg_keep = df_neg.sample(
        frac=0.15,  # por ejemplo, quedarte con el 15% de negativos
        random_state=42
    )

    df_train_bal = pd.concat([df_pos, neg_keep])

    X_train = df_train_bal.drop(columns=[TARGET_COL])
    y_train = df_train_bal[TARGET_COL].values


    print(
        f"Splits preparados - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    run_name = f"xgb_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        if batch_id is not None:
            mlflow.set_tag("batch_id", batch_id)
            mlflow.log_param("batch_id", batch_id)

        preprocessor = make_preprocessor(X_train)
        xgb_model = _make_xgb_model(y_train)

        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb_model),
        ])

        clf.fit(
            X_train,
            y_train,
            model__eval_set=[(X_val, y_val)],
            model__early_stopping_rounds=30,
            model__verbose=False,
        )

        proba_val = clf.predict_proba(X_val)[:, 1]
        proba_test = clf.predict_proba(X_test)[:, 1]

        metrics_val = _compute_metrics(y_val, proba_val, threshold=0.5)
        metrics_test = _compute_metrics(y_test, proba_test, threshold=0.5)

        hyperparams = xgb_model.get_params() if hasattr(xgb_model, "get_params") else {}
        mlflow.log_params(hyperparams)
        mlflow.log_params(
            {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
            }
        )

        for k, v in metrics_val.items():
            if k != "confusion_matrix":
                mlflow.log_metric(f"val_{k}", float(v))
        for k, v in metrics_test.items():
            if k != "confusion_matrix":
                mlflow.log_metric(f"test_{k}", float(v))

        plots_dir = ARTIFACTS_DIR / "plots"

        trained_xgb_model = clf.named_steps["model"]
        X_test_transformed = clf.named_steps["preprocessor"].transform(X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()

        try:
            feature_names_out = (
                clf.named_steps["preprocessor"].get_feature_names_out()
            )
            X_test_for_plots = pd.DataFrame(
                X_test_transformed, columns=feature_names_out
            )
        except Exception:
            X_test_for_plots = pd.DataFrame(X_test_transformed)

        interpretability_results = _generate_interpretability_plots(
            trained_xgb_model, X_test_for_plots, y_test, plots_dir
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)

        metrics = {
            "val": metrics_val,
            "test": metrics_test,
            "model_path": str(MODEL_PATH),
        }

        metrics_path = ARTIFACTS_DIR / "metrics_train.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        comprehensive_metrics = {
            "validation": metrics_val,
            "test": metrics_test,
            "model_path": str(MODEL_PATH),
        }

        tracking_path = _save_comprehensive_tracking(
            comprehensive_metrics,
            hyperparams,
            interpretability_results,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        mlflow.sklearn.log_model(clf, artifact_path="model")
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_pickle")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(tracking_path), artifact_path="tracking")

        for key, value in interpretability_results.items():
            if (
                isinstance(value, str)
                and value.endswith(".png")
                and os.path.exists(value)
            ):
                mlflow.log_artifact(value, artifact_path="plots")

        print("Métricas Validación:")
        for metric, value in metrics_val.items():
            if metric != "confusion_matrix":
                print(f"   • {metric}: {value:.4f}")

        print("Métricas Test:")
        for metric, value in metrics_test.items():
            if metric != "confusion_matrix":
                print(f"   • {metric}: {value:.4f}")

        print(f"Modelo guardado en: {MODEL_PATH}")
        print(f"Métricas básicas: {metrics_path}")
        print(f"Tracking completo: {tracking_path}")
        print(f"Gráficos en: {plots_dir}")
        print(f"MLflow run_id: {run.info.run_id}")
        print("=" * 60)

        return {
            **metrics,
            "comprehensive_tracking_path": tracking_path,
            "interpretability_results": interpretability_results,
            "mlflow_run_id": run.info.run_id,
        }


def train_model(batch_id: str | None = None):
    """
    Wrapper para usar desde el DAG de Airflow.
    """
    print(f"[TRAIN] train_model llamado con batch_id={batch_id}")
    return train_and_evaluate(batch_id=batch_id)
