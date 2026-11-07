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


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(snapshot_date_str,silver_loan_daily_directory,silver_attributes_directory,silver_financials_directory,
                                silver_clickstream_directory,gold_feature_store_directory,spark):
    # prepare arguments
    date_suffix = snapshot_date_str.replace('-', '_')
 
    # keep loans that START this month i.e. mob 0 
    lms_filepath = silver_loan_daily_directory + "silver_loan_daily_" + date_suffix + '.parquet'
    loans_df = spark.read.parquet(lms_filepath) \
        .filter(col("mob") == 0) \
        .select("loan_id", "Customer_ID", "loan_start_date", "tenure", "loan_amt")
    print('loans starting', snapshot_date_str, ':', loans_df.count())
 
    # load ustomer silver tables all months up to this date (nothing from the future is loaded)
    attr_df = spark.read.parquet(silver_attributes_directory + "silver_attributes_*.parquet") \
        .filter(col("snapshot_date") <= snapshot_date_str) \
        .select(col("Customer_ID").alias("attr_cid"), col("snapshot_date").alias("attr_date"),
                "Age_clean", "Occupation")
 
    fin_df = spark.read.parquet(silver_financials_directory + "silver_financials_*.parquet") \
        .filter(col("snapshot_date") <= snapshot_date_str) \
        .select(col("Customer_ID").alias("fin_cid"), col("snapshot_date").alias("fin_date"),
                "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts_clean",
                "Num_Credit_Card_clean", "Interest_Rate_clean", "Num_of_Loan_clean",
                "Outstanding_Debt", "Credit_Utilization_Ratio", "Credit_History_Age_in_mths",
                "delayed_payment_rate", "Monthly_Balance_clean",
                "Delay_from_due_date", "Num_Credit_Inquiries_clean", "Changed_Credit_Limit",
                "Total_EMI_per_month_clean", "Amount_invested_monthly",
                "Credit_Mix", "Payment_of_Min_Amount", "spent_level", "value_tier", "Type_of_Loan")

    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    click_df = spark.read.parquet(silver_clickstream_directory + "silver_clickstream_*.parquet") \
        .filter(col("snapshot_date") <= snapshot_date_str) \
        .select(col("Customer_ID").alias("click_cid"), col("snapshot_date").alias("click_date"), *fe_cols)
 
    # 3. Join on Customer_ID with the DATE AS A CONDITION: only data available at loan start
    gold_df = loans_df.join(attr_df,
        (col("Customer_ID") == col("attr_cid")) & (col("attr_date") <= col("loan_start_date")), "left")
 
    gold_df = gold_df.join(fin_df,
        (col("Customer_ID") == col("fin_cid")) & (col("fin_date") <= col("loan_start_date")), "left")

    # count of loan types e.g. "Auto Loan, and Payday Loan" -> 2 (no loans listed -> 0)
    gold_df = gold_df.withColumn("Type_of_Loan_count",
        F.when(col("Type_of_Loan").isNull(), 0)
         .otherwise(F.size(F.split(col("Type_of_Loan"), ","))).cast(IntegerType()))

    # 4. Clickstream: average of the 6 months up to loan start
    click_6m = loans_df.join(click_df,
        (col("Customer_ID") == col("click_cid")) &
        (col("click_date") <= col("loan_start_date")) &
        (col("click_date") > F.add_months(col("loan_start_date"), -6)), "left") \
        .groupBy("loan_id") \
        .agg(*[F.avg(c).alias(c + "_avg_6m") for c in fe_cols])
 
    gold_df = gold_df.join(click_6m, on="loan_id", how="left")
 
    # 5. Tidy up: drop renamed keys, tag partition date
    gold_df = gold_df.drop("attr_cid", "fin_cid") \
        .withColumn("snapshot_date", F.lit(snapshot_date_str).cast(DateType()))
 
    # 6. Save gold feature store table (one row per loan)
    partition_name = "gold_feature_store_" + date_suffix + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    gold_df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', gold_df.count())
 
    return gold_df
