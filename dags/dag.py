from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.python import PythonSensor
from airflow.models import DagRun
import pendulum

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

from model_config import MODEL_NAMES, MODEL_TRAIN_DATE   # train date 2024-05-01, xgboost + logistic_regression
# new applications are scored every day by the daily_inference DAG (dags/dag_daily_inference.py)
DAILY_INFERENCE_DAG_ID = "daily_inference"


def daily_inference_done(ds, **_):
    # monitoring of a month needs that day's scores; before deployment there is nothing to wait for
    if ds < MODEL_TRAIN_DATE:
        return True
    runs = DagRun.find(dag_id=DAILY_INFERENCE_DAG_ID, execution_date=pendulum.parse(ds))
    return bool(runs) and runs[0].get_state() == "success"


def run(script, args=""):
    # every task runs from the project root so utils/, data/, datamart/ and model_bank/ resolve
    return f'cd /app && python3 scripts/{script} --snapshotdate "{{{{ ds }}}}" {args}'


with DAG(
    'dag',
    default_args=default_args,
    description='monthly: data pipeline, model training and monitoring (new applications are scored daily in daily_inference)',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),   # last scored loans (2024-06) get their labels in 2024-12
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,
) as dag:

    # --- label store ---
    dep_check_source_label_data = EmptyOperator(task_id="dep_check_source_label_data")

    bronze_lms = BashOperator(task_id="bronze_lms", bash_command=run("data_bronze.py", "--table lms"))
    silver_lms = BashOperator(task_id="silver_lms", bash_command=run("data_silver.py", "--table lms"))
    gold_label_store = BashOperator(task_id="gold_label_store", bash_command=run("data_gold.py", "--store label"))

    label_store_completed = EmptyOperator(task_id="label_store_completed")

    dep_check_source_label_data >> bronze_lms >> silver_lms >> gold_label_store >> label_store_completed

    # --- feature store ---
    gold_feature_store = BashOperator(task_id="gold_feature_store", bash_command=run("data_gold.py", "--store feature"))
    feature_store_completed = EmptyOperator(task_id="feature_store_completed")

    for table in ["attributes", "financials", "clickstream"]:
        dep_check = EmptyOperator(task_id=f"dep_check_source_data_{table}")
        bronze = BashOperator(task_id=f"bronze_{table}", bash_command=run("data_bronze.py", f"--table {table}"))
        silver = BashOperator(task_id=f"silver_{table}", bash_command=run("data_silver.py", f"--table {table}"))
        dep_check >> bronze >> silver >> gold_feature_store

    # gold features start from the loans opened this month (silver loan_daily, mob 0)
    silver_lms >> gold_feature_store
    gold_feature_store >> feature_store_completed

    # --- model training (runs on the training date only; skips otherwise) ---
    model_train_start = EmptyOperator(task_id="model_train_start")
    model_train = BashOperator(task_id="model_train",
                               bash_command=run("model_train.py", f"--modeltraindate {MODEL_TRAIN_DATE}"))
    model_train_completed = EmptyOperator(task_id="model_train_completed")

    label_store_completed >> model_train_start
    feature_store_completed >> model_train_start
    model_train_start >> model_train >> model_train_completed

    # --- deployment scoring (champion and challenger): on the train date only, the deployed models also score
    # the 6 loan months between OOT and deployment, so production Gini shows before new loans' labels mature ---
    model_inference_deploy_start = EmptyOperator(task_id="model_inference_deploy_start")
    model_inference_deploy_completed = EmptyOperator(task_id="model_inference_deploy_completed")
    model_train_completed >> model_inference_deploy_start

    # --- production scoring happens in the daily_inference DAG; wait for today's (the 1st's) run ---
    wait_for_daily_inference = PythonSensor(task_id="wait_for_daily_inference", python_callable=daily_inference_done,
                                            mode="reschedule", poke_interval=60, timeout=6 * 60 * 60)

    # --- model monitoring (needs predictions and the labels known so far) ---
    model_monitor_start = EmptyOperator(task_id="model_monitor_start")
    model_monitor_completed = EmptyOperator(task_id="model_monitor_completed")
    model_inference_deploy_completed >> model_monitor_start
    wait_for_daily_inference >> model_monitor_start
    label_store_completed >> model_monitor_start

    for model_type, model_name in MODEL_NAMES.items():
        deploy = BashOperator(task_id=f"model_inference_deploy_{model_type}",
                              bash_command=run("model_inference.py", f"--modelname {model_name} --deploy"))
        monitor = BashOperator(task_id=f"model_monitor_{model_type}",
                               bash_command=run("model_monitoring.py", f"--modelname {model_name}"))
        model_inference_deploy_start >> deploy >> model_inference_deploy_completed
        model_monitor_start >> monitor >> model_monitor_completed

    # --- alerts: champion vs challenger chart; fails (red) if the champion breaches a threshold ---
    model_alert = BashOperator(task_id="model_alert", retries=0,
                               bash_command=run("model_alert.py", "--modelnames " + ",".join(MODEL_NAMES.values())))
    model_monitor_completed >> model_alert
