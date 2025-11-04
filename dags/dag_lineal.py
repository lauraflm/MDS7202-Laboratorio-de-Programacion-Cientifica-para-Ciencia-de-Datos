# dags/dag_lineal.py
from datetime import datetime
import os

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", "/opt/airflow")
RUNS_DIR = os.path.join(AIRFLOW_HOME, "runs")
DATA_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"

default_args = {"owner": "airflow"}

with DAG(
    dag_id="hiring_lineal",
    default_args=default_args,
    start_date=datetime(2024, 10, 1),
    schedule=None,          # ejecución manual
    catchup=False,          # sin backfill
    tags=["lab9", "lineal"],
) as dag:

    # 1) start_pipeline (placeholder)
    start_pipeline = EmptyOperator(task_id="start_pipeline")

    # 2) create_folders
    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
    )

    # 3) download_data -> guarda en /runs/{{ ds }}/raw/data_1.csv
    download_data = BashOperator(
        task_id="download_data",
        bash_command=(
            "mkdir -p {runs}/{{{{ ds }}}}/raw && "
            "curl -L -o {runs}/{{{{ ds }}}}/raw/data_1.csv {url}"
        ).format(runs=RUNS_DIR, url=DATA_URL),
    )

    # 4) split_data
    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
    )

    # 5) preprocess_and_train
    preprocess_and_train_task = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
    )

    # 6) gradio_interface (no bloquea el operador)
    gradio_interface_task = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
    )

    # encadenamiento DAG (siguiendo la ref lineal)
    (
        start_pipeline
        >> create_folders_task
        >> download_data
        >> split_data_task
        >> preprocess_and_train_task
        >> gradio_interface_task
    )
