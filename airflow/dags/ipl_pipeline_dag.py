from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Make sure Python can import your scripts
sys.path.append("/opt/airflow/scripts")

from ingest_data import ingest_data
from preprocess_data import preprocess
from train_model import train_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ipl_pipeline_dag',
    default_args=default_args,
    description='IPL Win Predictor ML Pipeline',
    schedule_interval=None,
    catchup=False
) as dag:

    ingest = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
        do_xcom_push=False,        # disable XCom
    )

    preprocess = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess,
        do_xcom_push=False,        # disable XCom
    )

    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        do_xcom_push=False,        # disable XCom
    )

    ingest >> preprocess >> train

