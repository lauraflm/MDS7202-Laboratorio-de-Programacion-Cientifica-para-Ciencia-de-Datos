from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule


# ================================================================
# Helpers
# ================================================================

def _ensure_path():
    """
    Asegura que /opt/airflow esté en sys.path
    para poder hacer 'import src.*' dentro de las tasks.
    """
    import sys
    root = "/opt/airflow"
    if root not in sys.path:
        sys.path.append(root)


def _get_batch_id(context):
    """
    Obtiene el batch_id de forma robusta desde dag_run.conf.
    """
    dag_run = context.get("dag_run")
    if dag_run is not None and dag_run.conf is not None:
        conf = dag_run.conf
        if "batch_id" in conf and conf["batch_id"]:
            return str(conf["batch_id"])

    # Si no se pasa un batch_id, usa la fecha lógica
    logical_date = context["logical_date"]
    return logical_date.strftime("%Y%m%d")



# ================================================================
# Wrappers para llamar a la lógica en src/*
# ================================================================

def _build_weekly_features_wrapper(**context):
    """
    Construye features para el batch de la corrida actual.
    Usa batch_id proveniente de dag_run.conf (si existe)
    o logical_date por defecto.
    """
    _ensure_path()
    from src.preprocessing import build_weekly_features

    batch_id = _get_batch_id(context)
    print(f"[DAG] build_weekly_features para batch_id={batch_id}")
    return build_weekly_features(batch_id=batch_id)


def _data_quality_wrapper(**context):
    """
    Revisa calidad del batch actual (nulos, filas, etc.) y loguea en MLflow.
    """
    _ensure_path()
    from src.drift import check_data_quality

    batch_id = _get_batch_id(context)
    print(f"[DAG] check_data_quality para batch_id={batch_id}")
    return check_data_quality(batch_id=batch_id)


def _data_drift_wrapper(**context):
    """
    Calcula data drift del batch actual vs. dataset base y loguea en MLflow.
    """
    _ensure_path()
    from src.drift import check_data_drift

    batch_id = _get_batch_id(context)
    print(f"[DAG] check_data_drift para batch_id={batch_id}")
    return check_data_drift(batch_id=batch_id)


def _branch_drift_wrapper(**context):
    """
    Decide si reentrenar o no según los resultados de drift / performance.

    Debe retornar el task_id de la siguiente tarea:
      - "train_model"  -> si hay que reentrenar
      - "skip_train"   -> si NO hay que reentrenar
    """
    _ensure_path()
    from src.drift import decide_retraining

    batch_id = _get_batch_id(context)
    print(f"[DAG] decide_retraining para batch_id={batch_id}")

    next_task_id = decide_retraining(batch_id=batch_id)
    # Por seguridad, si algo raro pasa, fuerza train
    if next_task_id not in ["train_model", "skip_train"]:
        next_task_id = "train_model"
    return next_task_id


def _train_model_wrapper(**context):
    """
    Entrena / reentrena el modelo y actualiza el modelo productivo.
    """
    _ensure_path()
    from src.training import train_model

    batch_id = _get_batch_id(context)
    print(f"[DAG] train_model para batch_id={batch_id}")
    return train_model(batch_id=batch_id)


def _generate_predictions_wrapper(**context):
    _ensure_path()
    from src.predict import generate_predictions

    batch_id = _get_batch_id(context)
    print(f"[DAG] generate_predictions para batch_id={batch_id}")
    generate_predictions(batch_id=batch_id)
    # sin return


# ================================================================
# Definición del DAG
# ================================================================

default_args = {
    "owner": "grupo_sodai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sodai_pipeline_dag",
    description="Pipeline productivo SodAI Drinks - Entrega 3",
    default_args=default_args,
    start_date=datetime(2025, 12, 2),  # Fecha de inicio para Predicción 2
    schedule_interval="@weekly",       # El pipeline se ejecutará cada semana
    catchup=False,
    max_active_runs=1,
    tags=["sodai", "mlops", "entrega3"],
    ) as dag:

    start = EmptyOperator(task_id="start")

    t_build_features = PythonOperator(
        task_id="build_weekly_features",
        python_callable=_build_weekly_features_wrapper,
    )

    t_check_quality = PythonOperator(
        task_id="check_data_quality",
        python_callable=_data_quality_wrapper,
    )

    t_check_drift = PythonOperator(
        task_id="check_data_drift",
        python_callable=_data_drift_wrapper,
    )

    t_branch_drift = BranchPythonOperator(
        task_id="branch_drift",
        python_callable=_branch_drift_wrapper,
    )

    t_train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model_wrapper,
    )

    t_skip_train = EmptyOperator(task_id="skip_train")

    join_after_train = EmptyOperator(
        task_id="join_after_train",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_generate_predictions = PythonOperator(
        task_id="generate_predictions",
        python_callable=_generate_predictions_wrapper,
    )

    end = EmptyOperator(task_id="end")

    # Dependencias
    start >> t_build_features >> t_check_quality >> t_check_drift >> t_branch_drift
    t_branch_drift >> t_train_model >> join_after_train
    t_branch_drift >> t_skip_train >> join_after_train
    join_after_train >> t_generate_predictions >> end
