from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule



def _ensure_path():
    """
    Asegura que /opt/airflow esté en sys.path
    para poder hacer 'import src.*' dentro de las tasks.
    """
    import sys
    root = "/opt/airflow"
    if root not in sys.path:
        sys.path.append(root)


def _build_weekly_features_wrapper():
    _ensure_path()
    from src.preprocessing import build_weekly_features
    return build_weekly_features()


def _train_model_wrapper():
    _ensure_path()
    from src.training import train_model
    return train_model()


def _generate_predictions_wrapper():
    _ensure_path()
    from src.predict import generate_predictions
    return generate_predictions()


def _branch_drift_wrapper():
    _ensure_path()
    # Versión para pruebas: forzar siempre el entrenamiento
    return "train_model"

# ================================================================

default_args = {
    "owner": "grupo_sodai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sodai_pipeline_dag",
    description="Pipeline productivo SodAI Drinks - Entrega 2",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["sodai", "mlops", "entrega2"],
) as dag:

    start = EmptyOperator(task_id="start")

    t_build_features = PythonOperator(
        task_id="build_weekly_features",
        python_callable=_build_weekly_features_wrapper,
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

    start >> t_build_features >> t_branch_drift
    t_branch_drift >> t_train_model >> join_after_train
    t_branch_drift >> t_skip_train >> join_after_train
    join_after_train >> t_generate_predictions >> end
