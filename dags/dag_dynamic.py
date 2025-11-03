# dags/dag_dynamic.py
from datetime import datetime
import os

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

from hiring_dynamic_functions import (
    create_folders,
    load_and_merge,
    split_data,
    train_model,
    evaluate_and_select,
    gradio_best_interface,
)

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
RUNS_DIR = os.path.join(AIRFLOW_HOME, "runs")

DATA_URL_1 = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
DATA_URL_2 = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"

default_args = {"owner": "airflow"}

with DAG(
    dag_id="hiring_dynamic",
    default_args=default_args,
    start_date=datetime(2024, 10, 1),
    schedule="0 9 * * 1",   # ► Programado: todos los Lunes 09:00 UTC (ajusta si quieres)
    catchup=False,          # NO backfill
    tags=["lab9", "dynamic", "parallel"],
    doc_md="""
    ### Hiring Dynamic
    Descarga data_1 y opcionalmente data_2, merge → split → entrena RF/LogReg/GB en paralelo,
    evalúa accuracy y deja best_model.joblib. Sirve Gradio con el mejor.
    """
) as dag:

    start = EmptyOperator(task_id="start")

    mk_dirs = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
    )

    # Descargas en paralelo; si una falla, seguimos con la que llegue
    dl1 = BashOperator(
        task_id="download_data_1",
        bash_command=(
            "mkdir -p {runs}/{{{{ ds }}}}/raw && "
            "curl -L -f -o {runs}/{{{{ ds }}}}/raw/data_1.csv {u1}"
        ).format(runs=RUNS_DIR, u1=DATA_URL_1),
    )
    dl2 = BashOperator(
        task_id="download_data_2",
        bash_command=(
            "mkdir -p {runs}/{{{{ ds }}}}/raw && "
            # -f: si falla (404) no deja archivo vacío; el merge sabrá usar lo que exista
            "curl -L -f -o {runs}/{{{{ ds }}}}/raw/data_2.csv {u2} || true"
        ).format(runs=RUNS_DIR, u2=DATA_URL_2),
    )

    merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # basta con que al menos una descarga exista
    )

    do_split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
    )

    # Entrenamientos en paralelo
    train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_model,
        op_kwargs={"model_name": "random_forest"},
    )
    train_lr = PythonOperator(
        task_id="train_logreg",
        python_callable=train_model,
        op_kwargs={"model_name": "logreg"},
    )
    train_gb = PythonOperator(
        task_id="train_gboost",
        python_callable=train_model,
        op_kwargs={"model_name": "gboost"},
    )

    evaluate = PythonOperator(
        task_id="evaluate_and_select",
        python_callable=evaluate_and_select,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # esperamos a que terminen los 3 entrenos
    )

    serve_gradio = PythonOperator(
        task_id="gradio_best_interface",
        python_callable=gradio_best_interface,
    )

    start >> mk_dirs >> [dl1, dl2] >> merge >> do_split >> [train_rf, train_lr, train_gb] >> evaluate >> serve_gradio

