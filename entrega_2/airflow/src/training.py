# src/training.py

import json
import os
from pathlib import Path
import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns   # NO disponible en la imagen de Airflow

import mlflow
import mlflow.sklearn  # para modelos tipo scikit-learn / XGBoost

# Guarda todo dentro del repo de Airflow
mlflow.set_tracking_uri("file:airflow/artifacts/mlruns")
mlflow.set_experiment("entrega2-proyecto")

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

# Para interpretabilidad
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Feature importance plots will be generated instead.")

from .config import MODEL_PATH, ARTIFACTS_DIR
from .data_io import load_features
from .preprocessing import (
    make_preprocessor,
    TARGET_COL,
    SPLIT_COL,
)

# ============================================================
# Helpers para tracking e interpretabilidad
# ============================================================

def _generate_interpretability_plots(model, X_test, y_test, plots_dir):
    """
    Genera gráficos de interpretabilidad del modelo.
    
    Args:
        model: Modelo entrenado (XGBoost)
        X_test: Features de test
        y_test: Target de test  
        plots_dir: Directorio donde guardar los plots
    
    Returns:
        dict: Información sobre los plots generados
    """
    interpretability_results = {}

    try:
        # Crear directorio si no existe
        os.makedirs(plots_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Feature Importance Plot (básico de XGBoost)
        # ------------------------------------------------------------------
        if hasattr(model, "feature_importances_"):
            feature_importance = model.feature_importances_
            feature_names = (
                X_test.columns
                if hasattr(X_test, "columns")
                else [f"feature_{i}" for i in range(X_test.shape[1])]
            )
        else:
            feature_importance = None
            feature_names = (
                X_test.columns
                if hasattr(X_test, "columns")
                else [f"feature_{i}" for i in range(X_test.shape[1])]
            )

        if feature_importance is not None:
            importance_df = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": feature_importance,
                    }
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
            interpretability_results["top_features"] = top_features.to_dict("records")[:10]

        # ------------------------------------------------------------------
        # 2. Análisis SHAP (si está disponible)
        # ------------------------------------------------------------------
        if SHAP_AVAILABLE:
            try:
                explainer = shap.TreeExplainer(model)

                sample_size = min(200, len(X_test))
                X_sample = (
                    X_test.iloc[:sample_size]
                    if hasattr(X_test, "iloc")
                    else X_test[:sample_size]
                )
                y_sample = (
                    y_test[:sample_size]
                    if hasattr(y_test, "__getitem__")
                    else y_test[:sample_size]
                )

                shap_values = explainer.shap_values(X_sample)

                # A. Summary plot (beeswarm)
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    show=False,
                    max_display=20,
                )
                plt.title("SHAP Summary Plot - Feature Impact on Model Output")
                plt.tight_layout()
                shap_summary_path = os.path.join(
                    plots_dir, "shap_summary_beeswarm.png"
                )
                plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
                plt.close()

                # B. Summary bar plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    plot_type="bar",
                    feature_names=feature_names,
                    show=False,
                    max_display=15,
                )
                plt.title(
                    "SHAP Summary Plot - Mean Absolute Feature Importance"
                )
                plt.tight_layout()
                shap_bar_path = os.path.join(
                    plots_dir, "shap_summary_bar.png"
                )
                plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
                plt.close()

                # C. Waterfall plots
                positive_indices = np.where(
                    (explainer.expected_value + shap_values.sum(axis=1)) > 0.5
                )[0]
                if len(positive_indices) > 0:
                    idx_positive = positive_indices[0]
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[idx_positive],
                            base_values=explainer.expected_value,
                            data=(
                                X_sample.iloc[idx_positive]
                                if hasattr(X_sample, "iloc")
                                else X_sample[idx_positive]
                            ),
                            feature_names=feature_names,
                        ),
                        show=False,
                    )
                    plt.title(
                        f"SHAP Waterfall Plot - Positive Prediction Example (Sample {idx_positive})"
                    )
                    plt.tight_layout()
                    waterfall_pos_path = os.path.join(
                        plots_dir, "shap_waterfall_positive.png"
                    )
                    plt.savefig(waterfall_pos_path, dpi=300, bbox_inches="tight")
                    plt.close()
                else:
                    waterfall_pos_path = None

                negative_indices = np.where(
                    (explainer.expected_value + shap_values.sum(axis=1)) <= 0.5
                )[0]
                if len(negative_indices) > 0:
                    idx_negative = negative_indices[0]
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values[idx_negative],
                            base_values=explainer.expected_value,
                            data=(
                                X_sample.iloc[idx_negative]
                                if hasattr(X_sample, "iloc")
                                else X_sample[idx_negative]
                            ),
                            feature_names=feature_names,
                        ),
                        show=False,
                    )
                    plt.title(
                        f"SHAP Waterfall Plot - Negative Prediction Example (Sample {idx_negative})"
                    )
                    plt.tight_layout()
                    waterfall_neg_path = os.path.join(
                        plots_dir, "shap_waterfall_negative.png"
                    )
                    plt.savefig(waterfall_neg_path, dpi=300, bbox_inches="tight")
                    plt.close()
                else:
                    waterfall_neg_path = None

                # D. Force plot para múltiples ejemplos
                if len(X_sample) >= 5:
                    try:
                        plt.figure(figsize=(14, 8))
                        indices = np.linspace(
                            0, len(X_sample) - 1, 5, dtype=int
                        )
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[indices],
                            (
                                X_sample.iloc[indices]
                                if hasattr(X_sample, "iloc")
                                else X_sample[indices]
                            ),
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False,
                        )
                        plt.title("SHAP Force Plot - Multiple Examples")
                        plt.tight_layout()
                        force_plot_path = os.path.join(
                            plots_dir, "shap_force_plot.png"
                        )
                        plt.savefig(force_plot_path, dpi=300, bbox_inches="tight")
                        plt.close()
                    except Exception as e:
                        print(f"Force plot failed: {e}")
                        force_plot_path = None
                else:
                    force_plot_path = None

                # E. Partial Dependence Plots para top features
                try:
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    top_feature_indices = np.argsort(mean_abs_shap)[-3:]

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    for i, feature_idx in enumerate(top_feature_indices):
                        feature_name = feature_names[feature_idx]
                        shap.partial_dependence_plot(
                            feature_idx,
                            explainer.model,
                            X_sample,
                            ice=False,
                            model_expected_value=True,
                            feature_expected_value=True,
                            ax=axes[i],
                            show=False,
                        )
                        axes[i].set_title(
                            f"Partial Dependence: {feature_name}"
                        )

                    plt.suptitle("SHAP Partial Dependence Plots - Top 3 Features")
                    plt.tight_layout()
                    partial_dep_path = os.path.join(
                        plots_dir, "shap_partial_dependence.png"
                    )
                    plt.savefig(partial_dep_path, dpi=300, bbox_inches="tight")
                    plt.close()
                except Exception as e:
                    print(f"Partial dependence plots failed: {e}")
                    partial_dep_path = None

                # Guardar info SHAP
                interpretability_results["shap_summary_beeswarm"] = shap_summary_path
                interpretability_results["shap_summary_bar"] = shap_bar_path
                if waterfall_pos_path:
                    interpretability_results[
                        "shap_waterfall_positive"
                    ] = waterfall_pos_path
                if waterfall_neg_path:
                    interpretability_results[
                        "shap_waterfall_negative"
                    ] = waterfall_neg_path
                if force_plot_path:
                    interpretability_results["shap_force_plot"] = force_plot_path
                if partial_dep_path:
                    interpretability_results[
                        "shap_partial_dependence"
                    ] = partial_dep_path

                mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
                shap_feature_importance = [
                    {
                        "feature": feature_names[i],
                        "mean_abs_shap": float(mean_abs_shap_values[i]),
                    }
                    for i in np.argsort(mean_abs_shap_values)[-10:][::-1]
                ]

                interpretability_results[
                    "shap_feature_importance"
                ] = shap_feature_importance
                interpretability_results["shap_expected_value"] = float(
                    explainer.expected_value
                )
                interpretability_results["shap_sample_size"] = sample_size
                interpretability_results["shap_status"] = "success"

            except Exception as e:
                interpretability_results["shap_status"] = f"failed: {str(e)}"
                print(f"Error en análisis SHAP: {e}")
        else:
            interpretability_results["shap_status"] = "shap_not_available"
            print("SHAP no disponible - solo Feature Importance básico")

        # ------------------------------------------------------------------
        # 3. Confusion Matrix Plot (sin seaborn)
        # ------------------------------------------------------------------
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

        confusion_matrix_path = os.path.join(
            plots_dir, "confusion_matrix.png"
        )
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
        plt.close()

        interpretability_results["confusion_matrix_plot"] = confusion_matrix_path
        interpretability_results["confusion_matrix"] = cm.tolist()
        interpretability_results["plots_generation"] = "success"

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
    Guarda información completa del experimento en formato JSON organizado.
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
        "feature_names": list(X_train.columns)
        if hasattr(X_train, "columns")
        else None,
        "class_distribution_train": {
            "class_0": int((y_train == 0).sum()),
            "class_1": int((y_train == 1).sum()),
            "class_balance_ratio": float((y_train == 1).sum() / len(y_train)),
        },
        "class_distribution_test": {
            "class_0": int((y_test == 0).sum()),
            "class_1": int((y_test == 1).sum()),
            "class_balance_ratio": float((y_test == 1).sum() / len(y_test)),
        },
    }

    comprehensive_tracking = {
        "experiment_info": experiment_info,
        "hyperparameters": hyperparams,
        "metrics": metrics,
        "data_info": data_info,
        "interpretability": interpretability_results,
        "system_info": {
            "tracking_version": "1.0",
            "tracking_method": "custom_json",
            "mlflow_used": True,
        },
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
    Crea el modelo XGBoost con hiperparámetros razonables y
    corrige desbalance usando scale_pos_weight.
    """
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
    Función principal de entrenamiento con tracking completo + MLflow.
    """
    df = load_features()

    X_train, y_train, X_val, y_val, X_test, y_test = _prepare_splits(df)

    print(
        f"Splits preparados - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
    )

    run_name = f"xgb_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        preprocessor = make_preprocessor(X_train)
        xgb_model = _make_xgb_model(y_train)

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", xgb_model),
            ]
        )

        clf.fit(X_train, y_train)

        proba_val = clf.predict_proba(X_val)[:, 1]
        proba_test = clf.predict_proba(X_test)[:, 1]

        metrics_val = _compute_metrics(y_val, proba_val, threshold=0.5)
        metrics_test = _compute_metrics(y_test, proba_test, threshold=0.5)

        if hasattr(xgb_model, "get_params"):
            hyperparams = xgb_model.get_params()
        else:
            hyperparams = {}

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


def train_model():
    """
    Función pensada para usar directamente en el DAG de Airflow.
    """
    return train_and_evaluate()
