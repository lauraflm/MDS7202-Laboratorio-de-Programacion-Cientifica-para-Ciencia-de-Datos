import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history as plot_hist_mpl,
    plot_param_importances as plot_imp_mpl,
    plot_slice as plot_slice_mpl
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt


import sys
import pandas as pd

import os, sys, pandas as pd

DATA_PATH = os.getenv("DATA_PATH", sys.argv[1] if len(sys.argv) > 1 else "/app/data/water_potability.csv")




def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    return best_model


def optimize_model():
    data = pd.read_csv(DATA_PATH)
    

    X = data.drop('Potability', axis=1)
    y = data['Potability']
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Tamaño del dataset: {len(data)} muestras")
    print(f"Training real: {len(X_tr)} muestras ({len(X_tr)/len(data)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} muestras ({len(X_val)/len(data)*100:.1f}%)")
    print(f"Test set: {len(X_test)} muestras ({len(X_test)/len(data)*100:.1f}%)")
    

    neg_count = (y_tr == 0).sum()
    pos_count = (y_tr == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Balance de clases - Ratio negativo/positivo: {scale_pos_weight:.2f}")
    

    experiment_name = "XGBoost_Water_Potability_Hyperparameter_Optimization"
    print(f"Creando experimento MLflow: {experiment_name}")
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experimento creado con ID: {experiment_id}")
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Usando experimento existente con ID: {experiment_id}")
    
    def objective(trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 
                                                     scale_pos_weight * 0.5, 
                                                     scale_pos_weight * 1.5),
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'logloss'
        }
        
        run_name = f"XGBoost_Trial_{trial.number}_lr_{params['learning_rate']:.3f}_depth_{params['max_depth']}_estimators_{params['n_estimators']}"
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            
            mlflow.log_params(params)
            params['early_stopping_rounds'] = 50
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            f1 = f1_score(y_val, y_pred, average='binary')
            aupr = average_precision_score(y_val, y_proba)
            mlflow.log_metric("valid_f1", f1)
            mlflow.log_metric("valid_aupr", aupr)
            mlflow.log_metric("accuracy", (y_pred == y_val).mean())
            mlflow.log_metric("best_iteration", getattr(model, "best_iteration", params['n_estimators']))
            mlflow.log_metric("trial_number", trial.number)
            mlflow.sklearn.log_model(model, "model") 
            print(f"Trial {trial.number}: F1={f1:.4f}, AUPR={aupr:.4f}, Best Iter={getattr(model, 'best_iteration', 'N/A')}")
            
        return f1
    
    
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    
    study = optuna.create_study(
        direction='maximize',
        study_name="XGBoost_Optimization",
        sampler=sampler,
        pruner=pruner
    )
    study.optimize(objective, n_trials=50)
    
    print(f"\n{'='*60}")
    print(f"Optimización completada!")
    print(f"Mejor F1-score en validación: {study.best_value:.4f}")
    print(f"Mejores parámetros:")
    for param, value in study.best_params.items():
        print(f"  - {param}: {value}")
    print(f"{'='*60}\n")
    

    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    

    with mlflow.start_run(experiment_id=experiment_id, run_name="Optuna_Visualization_Plots"):
            
            
            fig1 = plot_hist_mpl(study)
            fig1.figure.savefig("plots/optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close(fig1.figure)
            mlflow.log_artifact("plots/optimization_history.png", "plots")
            
            
            fig2 = plot_imp_mpl(study)
            fig2.figure.savefig("plots/param_importances.png", dpi=300, bbox_inches='tight')
            plt.close(fig2.figure)
            mlflow.log_artifact("plots/param_importances.png", "plots")
            

            fig3 = plot_slice_mpl(study)
            fig3.figure.savefig("plots/slice_plot.png", dpi=300, bbox_inches='tight')
            plt.close(fig3.figure)
            mlflow.log_artifact("plots/slice_plot.png", "plots")
            
    best_model = get_best_model(experiment_id)
    
    model_path = 'models/best_xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_f1 = f1_score(y_test, y_test_pred, average='binary')
    test_aupr = average_precision_score(y_test, y_test_proba)
    test_acc = (y_test_pred == y_test).mean()
    
    print(f"Test F1-score: {test_f1:.4f}")
    print(f"Test AUPR: {test_aupr:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*60 + "\n")
    
    plt.figure(figsize=(12, 8))
    

    booster = best_model.get_booster()
    importance_dict = booster.get_score(importance_type='gain')
    

    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=True)
    
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Importancia de Variables (Gain) - Mejor Modelo XGBoost\n(Optimizado con Optuna)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Importancia (Gain)', fontsize=12)
    plt.ylabel('Variables', fontsize=12)
    plt.tight_layout()
    
    importance_plot_path = 'plots/feature_importance_best_model.png'
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    config_text = f"""
CONFIGURACIÓN DEL MODELO FINAL - XGBOOST OPTIMIZADO
==================================================

INFORMACIÓN DEL DATASET:
- Archivo: water_potability.csv
- Total de muestras: {len(data)}
- Número de características: {X.shape[1]}
- Características: {list(X.columns)}
- Balance de clases: {dict(y.value_counts())}
- Valores faltantes procesados: Sí (imputación por mediana)
- Scale pos weight calculado: {scale_pos_weight:.2f}

DIVISIÓN DE DATOS:
- Training real: {len(X_tr)} muestras ({len(X_tr)/len(data)*100:.1f}%)
- Validation set: {len(X_val)} muestras ({len(X_val)/len(data)*100:.1f}%)
- Test set: {len(X_test)} muestras ({len(X_test)/len(data)*100:.1f}%)
- Estratificación: Sí en todos los splits
- Random state: 42

- Mejor F1-score en validación: {study.best_value:.4f}

MÉTRICAS EN TEST SET (EVALUACIÓN FINAL - DATOS NO VISTOS):
- Test F1-score: {test_f1:.4f}
- Test AUPR: {test_aupr:.4f}
- Test Accuracy: {test_acc:.4f}

MEJORES HIPERPARÁMETROS:
{chr(10).join([f"- {param}: {value}" for param, value in study.best_params.items()])}

EXPERIMENTO MLFLOW:
- Nombre: {experiment_name}
- ID: {experiment_id}
- Métrica registrada: valid_f1

ARCHIVOS GENERADOS:
- Modelo serializado: {model_path}
- Gráficos de Optuna: plots/optimization_history.png, plots/param_importances.png, plots/slice_plot.png
- Importancia de variables: {importance_plot_path}
- Configuración: plots/model_final_configuration.txt
- Versiones de librerías: plots/library_versions.txt

VERSIONES DE LIBRERÍAS:
- pandas: {pd.__version__}
- numpy: {np.__version__}
- scikit-learn: {__import__('sklearn').__version__}
- xgboost: {xgb.__version__}
- optuna: {optuna.__version__}
- mlflow: {mlflow.__version__}
- matplotlib: {__import__('matplotlib').__version__}


FECHA DE CREACIÓN: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    config_path = 'plots/model_final_configuration.txt'
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_text)
    

    with mlflow.start_run(experiment_id=experiment_id, run_name="Final_Model_Summary_and_Artifacts"):

        mlflow.log_params(study.best_params)
        mlflow.log_metric("final_best_f1_validation", study.best_value)
        mlflow.log_metric("final_test_f1", test_f1)
        mlflow.log_metric("final_test_aupr", test_aupr)
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("total_trials", len(study.trials))
        

        mlflow.log_artifact(importance_plot_path, "plots")
        mlflow.log_artifact(config_path, "plots")
        mlflow.log_artifact(model_path, "models")
        

        requirements_content = f"""# Versiones de librerías utilizadas en el desarrollo
                                        pandas=={pd.__version__}
                                        numpy=={np.__version__}
                                        scikit-learn=={__import__('sklearn').__version__}
                                        xgboost=={xgb.__version__}
                                        optuna=={optuna.__version__}
                                        mlflow=={mlflow.__version__}
                                        matplotlib=={__import__('matplotlib').__version__}
                                        """
        
        with open('plots/library_versions.txt', 'w') as f:
            f.write(requirements_content)
        
        mlflow.log_artifact('plots/library_versions.txt', "plots")
    
    

    return best_model


if __name__ == "__main__":
    optimize_model()