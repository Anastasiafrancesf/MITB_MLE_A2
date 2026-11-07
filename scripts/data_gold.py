import argparse
import os

import pyspark

import utils.data_processing_gold_table as gold

# to call this script: python scripts/data_gold.py --snapshotdate "2023-01-01" --store label

silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"
silver_clickstream_directory = "datamart/silver/clickstream/"
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"


def main(snapshotdate, store):
    print('\n\n---starting job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    if store == "label":
        os.makedirs(gold_label_store_directory, exist_ok=True)
        gold.process_labels_gold_table(snapshotdate, silver_loan_daily_directory, gold_label_store_directory,
                                       spark, dpd=30, mob=6)
    else:
        os.makedirs(gold_feature_store_directory, exist_ok=True)
        gold.process_features_gold_table(snapshotdate, silver_loan_daily_directory, silver_attributes_directory,
                                         silver_financials_directory, silver_clickstream_directory,
                                         gold_feature_store_directory, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--store", type=str, required=True, choices=["label", "feature"])
    args = parser.parse_args()

    main(args.snapshotdate, args.store)
