import argparse
import os

import pyspark

import utils.data_processing_bronze_table

# to call this script: python scripts/data_bronze.py --snapshotdate "2023-01-01" --table lms

# table -> (source csv, bronze table name, bronze directory)
TABLES = {
    "lms": ("data/lms_loan_daily.csv", "loan_daily", "datamart/bronze/lms/"),
    "clickstream": ("data/feature_clickstream.csv", "clickstream", "datamart/bronze/clickstream/"),
    "attributes": ("data/features_attributes.csv", "attributes", "datamart/bronze/attributes/"),
    "financials": ("data/features_financials.csv", "financials", "datamart/bronze/financials/"),
}


def main(snapshotdate, table):
    print('\n\n---starting job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    file_path, table_name, bronze_directory = TABLES[table]
    os.makedirs(bronze_directory, exist_ok=True)

    utils.data_processing_bronze_table.process_bronze_table(file_path, table_name, snapshotdate, bronze_directory, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--table", type=str, required=True, choices=TABLES.keys())
    args = parser.parse_args()

    main(args.snapshotdate, args.table)
