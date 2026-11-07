import argparse
import os
import pickle
import pprint
from datetime import datetime

import pandas as pd
import pyspark
from dateutil.relativedelta import relativedelta

import utils.model_bank

# daily scoring (daily_inference DAG): scores the applications made on --snapshotdate
#   python scripts/model_inference.py --snapshotdate "2024-05-01" --modelname "credit_model_2024_05_01_xgboost.pkl"
# deploy (monthly DAG, deploy date only): also scores the 6 loan months between OOT and deployment
#   python scripts/model_inference.py --snapshotdate "2024-05-01" --modelname "credit_model_2024_05_01_xgboost.pkl" --deploy

# labels are observed 6 months after loan start, so the 6 loan months before the train date are unseen by training
LABEL_MATURITY_MONTHS = 6


def main(snapshotdate, modelname, deploy=False):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["model_name"] = modelname
    config["model_artefact_filepath"] = "model_bank/" + modelname
    config["predictions_directory"] = "datamart/gold/model_predictions/" + modelname[:-4] + "/"
    pprint.pprint(config)

    # --- load model artefact from model bank ---
    if not os.path.exists(config["model_artefact_filepath"]):
        print("skip: model not trained yet")
        return
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    print("Model loaded successfully! " + config["model_artefact_filepath"])

    # the model scores new applications from the day it was trained (deployed) onwards
    deploy_date_str = model_artefact["data_dates"]["model_train_date_str"]
    if snapshotdate < deploy_date_str:
        print(f"skip: model not deployed yet (deployed {deploy_date_str})")
        return

    if deploy:
        # on the deploy date, also score the loans that started after OOT but before deployment:
        # their labels had not matured at training, so they are unseen and show production performance
        # from 2024-11 onwards instead of waiting for the first post-deployment labels
        if snapshotdate != deploy_date_str:
            print(f"skip: pre-deployment loans are scored on the deploy date only ({deploy_date_str})")
            return
        deploy_date = datetime.strptime(deploy_date_str, "%Y-%m-%d")
        days = [(deploy_date - relativedelta(months=k)).strftime("%Y-%m-%d") for k in range(LABEL_MATURITY_MONTHS, 0, -1)]
        jobs = [(month, month, None, "production") for month in days]   # whole loan month, scored on the deploy date
    else:
        # daily: the applications made today sit in this month's feature store partition
        month = snapshotdate[:8] + "01"
        feature_store_filepath = "datamart/gold/feature_store/gold_feature_store_" + month.replace('-', '_') + ".parquet"
        if not os.path.exists(feature_store_filepath):
            raise FileNotFoundError(f"feature store for {month} not built yet: {feature_store_filepath}")
        # quick check before starting Spark: most days have no new applications
        start_dates = pd.read_parquet(feature_store_filepath, columns=["loan_start_date"])["loan_start_date"].astype(str)
        n_today = int((start_dates == snapshotdate).sum())
        print(f"{snapshotdate} applications today:", n_today)
        if n_today == 0:
            print("skip: no new applications to score today")
            return
        jobs = [(snapshotdate, month, snapshotdate, "production")]

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    for partition_date, month, day, period in jobs:
        score(spark, partition_date, month, day, period, model_artefact, config)

    spark.stop()
    print('\n\n---completed job---\n\n')


def score(spark, partition_date, month, day, period, model_artefact, config):
    modelname = config["model_name"]

    # --- load feature store: loans starting this month (and on this day, for daily scoring) ---
    feature_store_filepath = "datamart/gold/feature_store/gold_feature_store_" + month.replace('-', '_') + ".parquet"
    features_pdf = spark.read.parquet(feature_store_filepath).toPandas()
    if day is not None:
        features_pdf = features_pdf[features_pdf["loan_start_date"].astype(str) == day]
    print(f"{partition_date} ({period}) loans to score:", len(features_pdf))
    if features_pdf.empty:
        print("skip: no new loans to score")
        return

    # the model needs clickstream; applicants not in the clickstream feed are not scored (would need a fallback model)
    click_cols = [c for c in model_artefact["feature_cols"] if c.startswith("fe_")]
    has_click = features_pdf[click_cols].notna().all(axis=1)
    features_pdf = features_pdf[has_click]
    print("loans scored:", int(has_click.sum()), "| not scored (no clickstream):", int((~has_click).sum()))
    if features_pdf.empty:
        print("skip: no loans with clickstream to score")
        return

    # --- model prediction inference (preprocessing is inside the pipeline) ---
    y_inference = model_artefact["model"].predict_proba(features_pdf[model_artefact["feature_cols"]])[:, 1]

    y_inference_pdf = features_pdf[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = modelname
    # champion predictions are the ones used for decisions; challenger predictions are for comparison only
    y_inference_pdf["role"] = utils.model_bank.get_role(modelname)
    # period: OOT (scored at training) or production (scored by the deployed model)
    y_inference_pdf["period"] = period
    y_inference_pdf["model_predictions"] = y_inference

    # --- save model inference to datamart gold table ---
    os.makedirs(config["predictions_directory"], exist_ok=True)
    partition_name = modelname[:-4] + "_predictions_" + partition_date.replace('-', '_') + ".parquet"
    filepath = config["predictions_directory"] + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    parser.add_argument("--deploy", action="store_true", help="on the deploy date, score the 6 loan months before deployment")
    args = parser.parse_args()

    main(args.snapshotdate, args.modelname, args.deploy)
