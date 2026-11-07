import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_bronze_table(file_path, table_name, snapshot_date_str, bronze_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load data
    df = spark.read.csv(file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + ' row count:', df.count())

    # save bronze table to datamart
    partition_name = f"bronze_{table_name}_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

# def process_bronze_table_overwrite_files(file_path, table_name, snapshot_date_str, bronze_directory, spark):

#     snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

#     df = spark.read.csv(file_path, header=True, inferSchema=True).filter(col('snapshot_date') <= snapshot_date)
#     print(table_name + ' row count:', df.count())

#     partition_name = f"bronze_{table_name}_" + snapshot_date_str.replace('-', '_') + '.csv'
#     filepath = bronze_directory + partition_name
#     df.toPandas().to_csv(filepath, index=False)
#     print('saved to:', filepath)

#     return df