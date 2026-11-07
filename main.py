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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
snapshot_date_str = "2023-01-01"

start_date_str = "2023-01-01"
end_date_str = "2024-12-01"

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"
bronze_click_directory = "datamart/bronze/clickstream/"
bronze_fa_directory = "datamart/bronze/attributes/"
bronze_financials_directory = "datamart/bronze/financials/"

for directory in [bronze_lms_directory, bronze_click_directory, bronze_fa_directory, bronze_financials_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

lms_file_path = r"data/lms_loan_daily.csv"
click_file_path = r"data/feature_clickstream.csv"
attr_file_path = r"data/features_attributes.csv"
financials_file_path = r"data/features_financials.csv"

# date-partitioned tables
partitioned_pairs = [
    (lms_file_path, "loan_daily", bronze_lms_directory),
    (click_file_path, "clickstream", bronze_click_directory),
    (attr_file_path, "attributes", bronze_fa_directory),
    (financials_file_path, "financials", bronze_financials_directory)
]

# run bronze backfill
for date_str in dates_str_lst:
    for file_path, table_name, bronze_directory in partitioned_pairs:
        utils.data_processing_bronze_table.process_bronze_table(
            file_path, table_name, date_str, bronze_directory, spark)

# create silver datalake directories
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_clickstream_directory = "datamart/silver/clickstream/"
silver_attributes_directory = "datamart/silver/attributes/"
silver_financials_directory = "datamart/silver/financials/"

for directory in [silver_loan_daily_directory, silver_clickstream_directory,
                   silver_attributes_directory, silver_financials_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# run silver backfill
for date_str in dates_str_lst:
    utils.data_processing_silver_table.process_silver_table_lms(
        date_str, bronze_lms_directory, silver_loan_daily_directory, spark
    )

    utils.data_processing_silver_table.process_silver_table_clickstream(
        date_str, bronze_click_directory, silver_clickstream_directory, spark
    )
    
    utils.data_processing_silver_table.process_silver_table_attributes(
        date_str, bronze_fa_directory, silver_attributes_directory, spark
    )
    
    utils.data_processing_silver_table.process_silver_table_financials(
        date_str, bronze_financials_directory, silver_financials_directory, spark
    )

# create gold datalake
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)
    utils.data_processing_gold_table.process_features_gold_table(date_str,silver_loan_daily_directory,silver_attributes_directory,silver_financials_directory,silver_clickstream_directory,gold_feature_store_directory,spark)


folder_path = gold_label_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

df.show()

folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df_feature = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",df.count())

folder_path = gold_feature_store_directory
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
df_feature = spark.read.parquet(*files_list)
print("row_count:", df_feature.count())

# 1. Identifiers & Loan info (LMS, known at application)
lms_cols = ['loan_id', 'Customer_ID', 'snapshot_date', 'loan_start_date', 'loan_amt', 'tenure']
df_feature.select(lms_cols).show(5)

# 2. Financial Profile (Financials)
fin1_cols = ['Customer_ID', 'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt']
df_feature.select(fin1_cols).show(5)

fin2_cols = ['Credit_Utilization_Ratio', 'Credit_History_Age_in_mths',
             'delayed_payment_rate', 'Monthly_Balance_clean']
df_feature.select(fin2_cols).show(5)

# 3. Demographics & Credit Accounts
attr1_cols = ['Customer_ID', 'Age_clean', 'Occupation', 'Num_Bank_Accounts_clean']
df_feature.select(attr1_cols).show(5)

attr2_cols = ['Num_Credit_Card_clean', 'Interest_Rate_clean', 'Num_of_Loan_clean']
df_feature.select(attr2_cols).show(5)

# 4. Clickstream: 6-month averages up to loan start
click_cols = ['Customer_ID', 'fe_1_avg_6m', 'fe_2_avg_6m', 'fe_3_avg_6m', 'fe_4_avg_6m', 'fe_5_avg_6m']
print("--- 4. Clickstream Sample ---")
df_feature.select(click_cols).show(5)
    