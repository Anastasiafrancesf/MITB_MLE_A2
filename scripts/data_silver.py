import argparse
import os

import pyspark

import utils.data_processing_silver_table as silver

# to call this script: python scripts/data_silver.py --snapshotdate "2023-01-01" --table lms

# table -> (silver function, bronze directory, silver directory)
TABLES = {
    "lms": (silver.process_silver_table_lms, "datamart/bronze/lms/", "datamart/silver/loan_daily/"),
    "clickstream": (silver.process_silver_table_clickstream, "datamart/bronze/clickstream/", "datamart/silver/clickstream/"),
    "attributes": (silver.process_silver_table_attributes, "datamart/bronze/attributes/", "datamart/silver/attributes/"),
    "financials": (silver.process_silver_table_financials, "datamart/bronze/financials/", "datamart/silver/financials/"),
}


def main(snapshotdate, table):
    print('\n\n---starting job---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    process, bronze_directory, silver_directory = TABLES[table]
    os.makedirs(silver_directory, exist_ok=True)

    process(snapshotdate, bronze_directory, silver_directory, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--table", type=str, required=True, choices=TABLES.keys())
    args = parser.parse_args()

    main(args.snapshotdate, args.table)
