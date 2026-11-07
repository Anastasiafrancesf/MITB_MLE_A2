from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor

from model_config import MODEL_NAMES, MODEL_TRAIN_DATE

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# daily scoring: each day, the champion and the challenger score the applications made that day.
# Training, deployment and monitoring stay in the monthly DAG ('dag'): ~500 loans a month is enough for a stable Gini,
# one day's applications are not. The two DAGs share the model bank (champion.json) and the gold predictions table.


def run(script, args=""):
    return f'cd /app && python3 scripts/{script} --snapshotdate "{{{{ ds }}}}" {args}'


with DAG(
    'daily_inference',
    default_args=default_args,
    description='daily: score the day\'s new applications with the champion and challenger',
    schedule_interval='0 0 * * *',  # every day at 00:00
    start_date=datetime.strptime(MODEL_TRAIN_DATE, "%Y-%m-%d"),   # scoring starts on the deploy date
    end_date=datetime(2024, 12, 1),   # same end as the monthly DAG
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,
) as dag:

    # the day's features come from this month's feature store, built by the monthly DAG on the 1st,
    # and the model must be in the model bank: wait for that month's run to finish training
    wait_for_monthly_features = ExternalTaskSensor(
        task_id="wait_for_monthly_features",
        external_dag_id="dag",
        external_task_id="model_train_completed",
        execution_date_fn=lambda dt, **_: dt.replace(day=1),
        mode="reschedule", poke_interval=60, timeout=6 * 60 * 60,
    )

    for model_type, model_name in MODEL_NAMES.items():
        inference = BashOperator(task_id=f"model_inference_{model_type}",
                                 bash_command=run("model_inference.py", f"--modelname {model_name}"))
        wait_for_monthly_features >> inference
