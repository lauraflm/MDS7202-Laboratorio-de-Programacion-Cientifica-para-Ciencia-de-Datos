# dags/dag_dynamic.py
from datetime import datetime, date
import os

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

from hiring_dynamic_functions import (
    create_folders,
    load_and_merge,
    split_data,
    train_model,
    evaluate_models,
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
    schedule="0 15 5 * *",   # cada 5 de mes a las 15:00
    catchup=True,          # si backfill
    tags=["lab9", "dynamic", "parallel"],
    doc_md="""
    ### Hiring Dynamic
    Descarga data_1 o data_2, dependiendo de la fecha, merge → split → entrena RF/LogReg/GB en paralelo,
    evalúa accuracy y deja best_model.joblib. Sirve Gradio con el mejor.
    """
) as dag:
    # marca posición inicial
    start = EmptyOperator(task_id="start")

    mk_dirs = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # 4) branching por fecha
    def choose_downloads(ds, **_):
        exec_date = datetime.strptime(ds, "%Y-%m-%d").date()
        cutoff = date(2024, 11, 1) # 1 de noviembre 2024
        if exec_date < cutoff:
            return "download_data_1"
        return ["download_data_1", "download_data_2"]

    branch = BranchPythonOperator(
        task_id="branch_downloads",
        python_callable=choose_downloads,
        op_kwargs={"ds": "{{ ds }}"},
    )

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
            "curl -L -f -o {runs}/{{{{ ds }}}}/raw/data_2.csv {u2}"
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
        python_callable=evaluate_models,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # esperamos a que terminen los 3 entrenos
    )


    start >> mk_dirs >> branch
    branch >> dl1
    branch >> dl2
    [dl1, dl2] >> merge >> do_split >> [train_rf, train_lr, train_gb] >> evaluate