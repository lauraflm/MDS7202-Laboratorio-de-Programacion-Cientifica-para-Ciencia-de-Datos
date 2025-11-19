# src/training.py

import json
import os
from pathlib import Path
import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # 1. Feature Importance Plot
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'feature_{i}' for i in range(X_test.shape[1])]
            
            # DataFrame para ordenar features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Plot top 15 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Feature Importance - XGBoost Model')
            plt.gca().invert_yaxis()
            
            # Agregar valores en las barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
            
            plt.tight_layout()
            feature_importance_path = os.path.join(plots_dir, 'feature_importance.png')
            plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            interpretability_results['feature_importance_plot'] = feature_importance_path
            interpretability_results['top_features'] = top_features.to_dict('records')[:10]
        
        # 2. SHAP Analysis
        if SHAP_AVAILABLE:
            try:
                
                
                # Crear explainer
                explainer = shap.TreeExplainer(model)
                
                # Calcular SHAP values para una muestra (optimizado para memoria)
                sample_size = min(200, len(X_test))  # Aumentamos muestra para mejor representatividad
                X_sample = X_test.iloc[:sample_size] if hasattr(X_test, 'iloc') else X_test[:sample_size]
                y_sample = y_test[:sample_size] if hasattr(y_test, '__getitem__') else y_test[:sample_size]
                
                shap_values = explainer.shap_values(X_sample)
                
                # A. Summary plot (beeswarm)
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X_sample, 
                                feature_names=feature_names, show=False, max_display=20)
                plt.title('SHAP Summary Plot - Feature Impact on Model Output')
                plt.tight_layout()
                shap_summary_path = os.path.join(plots_dir, 'shap_summary_beeswarm.png')
                plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # B. Summary bar plot (feature importance based on SHAP)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", 
                                feature_names=feature_names, show=False, max_display=15)
                plt.title('SHAP Summary Plot - Mean Absolute Feature Importance')
                plt.tight_layout()
                shap_bar_path = os.path.join(plots_dir, 'shap_summary_bar.png')
                plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # C. Waterfall plot 
                # Caso 1: Predicción positiva con alta confianza
                positive_indices = np.where((explainer.expected_value + shap_values.sum(axis=1)) > 0.5)[0]
                if len(positive_indices) > 0:
                    idx_positive = positive_indices[0]
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(
                        shap.Explanation(values=shap_values[idx_positive], 
                                       base_values=explainer.expected_value, 
                                       data=X_sample.iloc[idx_positive] if hasattr(X_sample, 'iloc') else X_sample[idx_positive],
                                       feature_names=feature_names),
                        show=False
                    )
                    plt.title(f'SHAP Waterfall Plot - Positive Prediction Example (Sample {idx_positive})')
                    plt.tight_layout()
                    waterfall_pos_path = os.path.join(plots_dir, 'shap_waterfall_positive.png')
                    plt.savefig(waterfall_pos_path, dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    waterfall_pos_path = None
                
                # Caso 2: Predicción negativa con alta confianza
                negative_indices = np.where((explainer.expected_value + shap_values.sum(axis=1)) <= 0.5)[0]
                if len(negative_indices) > 0:
                    idx_negative = negative_indices[0]
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(
                        shap.Explanation(values=shap_values[idx_negative], 
                                       base_values=explainer.expected_value, 
                                       data=X_sample.iloc[idx_negative] if hasattr(X_sample, 'iloc') else X_sample[idx_negative],
                                       feature_names=feature_names),
                        show=False
                    )
                    plt.title(f'SHAP Waterfall Plot - Negative Prediction Example (Sample {idx_negative})')
                    plt.tight_layout()
                    waterfall_neg_path = os.path.join(plots_dir, 'shap_waterfall_negative.png')
                    plt.savefig(waterfall_neg_path, dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    waterfall_neg_path = None
                
                # D. Force plot para múltiples ejemplos
                if len(X_sample) >= 5:
                    try:
                        plt.figure(figsize=(14, 8))
                        # Seleccionar 5 ejemplos diversos
                        indices = np.linspace(0, len(X_sample)-1, 5, dtype=int)
                        
                        shap.force_plot(explainer.expected_value, shap_values[indices], 
                                      X_sample.iloc[indices] if hasattr(X_sample, 'iloc') else X_sample[indices],
                                      feature_names=feature_names, matplotlib=True, show=False)
                        plt.title('SHAP Force Plot - Multiple Examples')
                        plt.tight_layout()
                        force_plot_path = os.path.join(plots_dir, 'shap_force_plot.png')
                        plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"Force plot failed: {e}")
                        force_plot_path = None
                else:
                    force_plot_path = None
                
                # E. Partial Dependence Plots para top features
                try:
                    # Calcular importancia media absoluta de SHAP
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    top_feature_indices = np.argsort(mean_abs_shap)[-3:]  # Top 3 features
                    
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    for i, feature_idx in enumerate(top_feature_indices):
                        feature_name = feature_names[feature_idx]
                        shap.partial_dependence_plot(
                            feature_idx, explainer.model, X_sample, ice=False,
                            model_expected_value=True, feature_expected_value=True,
                            ax=axes[i], show=False
                        )
                        axes[i].set_title(f'Partial Dependence: {feature_name}')
                    
                    plt.suptitle('SHAP Partial Dependence Plots - Top 3 Features')
                    plt.tight_layout()
                    partial_dep_path = os.path.join(plots_dir, 'shap_partial_dependence.png')
                    plt.savefig(partial_dep_path, dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Partial dependence plots failed: {e}")
                    partial_dep_path = None
                
                # Guardar información de SHAP
                interpretability_results['shap_summary_beeswarm'] = shap_summary_path
                interpretability_results['shap_summary_bar'] = shap_bar_path
                if waterfall_pos_path:
                    interpretability_results['shap_waterfall_positive'] = waterfall_pos_path
                if waterfall_neg_path:
                    interpretability_results['shap_waterfall_negative'] = waterfall_neg_path
                if force_plot_path:
                    interpretability_results['shap_force_plot'] = force_plot_path
                if partial_dep_path:
                    interpretability_results['shap_partial_dependence'] = partial_dep_path
                
                # Calcular y guardar estadísticas de SHAP
                mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
                shap_feature_importance = [
                    {"feature": feature_names[i], "mean_abs_shap": float(mean_abs_shap_values[i])}
                    for i in np.argsort(mean_abs_shap_values)[-10:][::-1]  # Top 10
                ]
                
                interpretability_results['shap_feature_importance'] = shap_feature_importance
                interpretability_results['shap_expected_value'] = float(explainer.expected_value)
                interpretability_results['shap_sample_size'] = sample_size
                interpretability_results['shap_status'] = "success"
                
                
            except Exception as e:
                interpretability_results['shap_status'] = f"failed: {str(e)}"
                print(f" Error en análisis SHAP: {e}")
        else:
            interpretability_results['shap_status'] = "shap_not_available"
            print(" SHAP no disponible - solo Feature Importance básico")
        
        # 3. Confusion Matrix Heatmap
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Compra', 'Compra'],
                   yticklabels=['No Compra', 'Compra'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        confusion_matrix_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        interpretability_results['confusion_matrix_plot'] = confusion_matrix_path
        interpretability_results['confusion_matrix'] = cm.tolist()
        
        interpretability_results['plots_generation'] = "success"
        
    except Exception as e:
        interpretability_results['plots_generation'] = f"failed: {str(e)}"
        print(f"Error generating interpretability plots: {e}")
    
    return interpretability_results


def _save_comprehensive_tracking(metrics, hyperparams, interpretability_results, 
                               X_train, y_train, X_test, y_test):
    """
    Guarda información completa del experimento en formato JSON organizado.
    
    Args:
        metrics: Métricas del modelo
        hyperparams: Hiperparámetros utilizados
        interpretability_results: Resultados de interpretabilidad
        X_train, y_train, X_test, y_test: Datos de entrenamiento y test
    
    Returns:
        str: Path al archivo de tracking guardado
    """
    
    # Información del experimento
    experiment_info = {
        'experiment_id': f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.datetime.now().isoformat(),
        'model_type': 'XGBoost',
        'model_version': 'v1.0',
        'data_version': 'weekly_features_v1'
    }
    
    # Información de los datos
    data_info = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
        'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else None,
        'class_distribution_train': {
            'class_0': int((y_train == 0).sum()),
            'class_1': int((y_train == 1).sum()),
            'class_balance_ratio': float((y_train == 1).sum() / len(y_train))
        },
        'class_distribution_test': {
            'class_0': int((y_test == 0).sum()),
            'class_1': int((y_test == 1).sum()),
            'class_balance_ratio': float((y_test == 1).sum() / len(y_test))
        }
    }
    
    # Estructura completa del tracking
    comprehensive_tracking = {
        'experiment_info': experiment_info,
        'hyperparameters': hyperparams,
        'metrics': metrics,
        'data_info': data_info,
        'interpretability': interpretability_results,
        'system_info': {
            'tracking_version': '1.0',
            'tracking_method': 'custom_json',
            'mlflow_used': False
        }
    }
    
    # Guardar el tracking completo
    tracking_path = ARTIFACTS_DIR / 'experiment_tracking.json'
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(tracking_path, 'w') as f:
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
    Función principal de entrenamiento con tracking completo:

    1. Carga las features desde parquet.
    2. Separa en train / val / test usando 'split'.
    3. Crea el preprocessor (ColumnTransformer) con make_preprocessor().
    4. Crea un Pipeline: preprocessor + XGBClassifier.
    5. Entrena con datos de train.
    6. Calcula métricas en val y test.
    7. Genera gráficos de interpretabilidad.
    8. Registra hiperparámetros y resultados completos.
    9. Guarda el pipeline completo en MODEL_PATH.
    10. Devuelve un dict con las métricas y rutas de artefactos.
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

    print(f"Splits preparados - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

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

    # 8) Obtener hiperparámetros del modelo entrenado
    if hasattr(xgb_model, 'get_params'):
        hyperparams = xgb_model.get_params()
    else:
        # Fallback con hiperparámetros conocidos
        hyperparams = {
            'n_estimators': getattr(xgb_model, 'n_estimators', 300),
            'learning_rate': getattr(xgb_model, 'learning_rate', 0.1),
            'max_depth': getattr(xgb_model, 'max_depth', 6),
            'min_child_weight': getattr(xgb_model, 'min_child_weight', 1.0),
            'subsample': getattr(xgb_model, 'subsample', 0.8),
            'colsample_bytree': getattr(xgb_model, 'colsample_bytree', 0.8),
            'gamma': getattr(xgb_model, 'gamma', 0.0),
            'scale_pos_weight': getattr(xgb_model, 'scale_pos_weight', 1.0),
            'reg_lambda': getattr(xgb_model, 'reg_lambda', 1.0),
            'reg_alpha': getattr(xgb_model, 'reg_alpha', 0.0),
            'random_state': getattr(xgb_model, 'random_state', 42),
            'objective': getattr(xgb_model, 'objective', 'binary:logistic'),
            'eval_metric': getattr(xgb_model, 'eval_metric', 'logloss'),
            'tree_method': getattr(xgb_model, 'tree_method', 'hist')
        }

    # 9) Generar gráficos de interpretabilidad
    plots_dir = ARTIFACTS_DIR / 'plots'
    
    # Obtener el modelo entrenado del pipeline para interpretabilidad
    trained_xgb_model = clf.named_steps['model']
    
    # Transformar X_test con el preprocessor para interpretabilidad
    X_test_transformed = clf.named_steps['preprocessor'].transform(X_test)
    
    # Convertir a DataFrame si es necesario (para nombres de features)
    if hasattr(X_test_transformed, 'toarray'):  # Es sparse matrix
        X_test_transformed = X_test_transformed.toarray()
    
    # Crear DataFrame con nombres de features si es posible
    try:
        feature_names_out = clf.named_steps['preprocessor'].get_feature_names_out()
        X_test_for_plots = pd.DataFrame(X_test_transformed, columns=feature_names_out)
    except:
        X_test_for_plots = pd.DataFrame(X_test_transformed)
    
    interpretability_results = _generate_interpretability_plots(
        trained_xgb_model, X_test_for_plots, y_test, plots_dir
    )

    # 10) Guardar modelo entrenado 
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    # 11) Guardar métricas básicas 
    metrics = {
        "val": metrics_val,
        "test": metrics_test,
        "model_path": str(MODEL_PATH),
    }

    metrics_path = ARTIFACTS_DIR / "metrics_train.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 12) Guardar tracking completo
    comprehensive_metrics = {
        'validation': metrics_val,
        'test': metrics_test,
        'model_path': str(MODEL_PATH)
    }
    
    tracking_path = _save_comprehensive_tracking(
        comprehensive_metrics, hyperparams, interpretability_results,
        X_train, y_train, X_test, y_test
    )

    # 13) Mostrar resumen por pantalla
    print(f"Métricas Validación:")
    for metric, value in metrics_val.items():
        if metric != 'confusion_matrix':
            print(f"   • {metric}: {value:.4f}")
    
    print(f"Métricas Test:")
    for metric, value in metrics_test.items():
        if metric != 'confusion_matrix':
            print(f"   • {metric}: {value:.4f}")
    
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas básicas: {metrics_path}")
    print(f"Tracking completo: {tracking_path}")
    print(f"Gráficos en: {plots_dir}")
    print("="*60)

    return {
        **metrics,
        'comprehensive_tracking_path': tracking_path,
        'interpretability_results': interpretability_results
    }


def train_model():
    """
    Función pensada para usar directamente en el DAG de Airflow.
    Solo llama a train_and_evaluate() y devuelve las métricas.
    """
    return train_and_evaluate()
