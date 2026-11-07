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

from pyspark.sql.functions import col, split, regexp_replace
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_table_lms(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_daily_directory, spark):
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes

    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    # add fe_1 through fe_20, all as FloatType
    column_type_map.update({f"fe_{i}": FloatType() for i in range(1, 21)})


    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # connect to bronze table
    partition_name = 'bronze_attributes_' + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.withColumn('Age', regexp_replace(col('Age'), "_", ""))

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    df = df.withColumn(
        "Occupation",
        F.when(col("Occupation") == "_______", None).otherwise(col("Occupation"))
    )
    df = df.withColumn(
        "SSN",
        F.when(col("SSN") == "#F%$D@*&8", None).otherwise(col("SSN"))
    )

    df = df.withColumn("Age_clean", col("Age").cast(IntegerType()))
    df = df.withColumn("Age_flag", (col("Age_clean") > 100) | (col("Age_clean") < 0))
    df = df.withColumn("Age_clean",
        F.when(col("Age_flag"), None).otherwise(col("Age_clean")))

    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # connect to bronze table
    partition_name = 'bronze_financials_'+ snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    underscore_cols = [
        "Annual_Income", "Outstanding_Debt", "Num_of_Loan", "Changed_Credit_Limit",
        "Num_of_Delayed_Payment", "Monthly_Balance"
    ]
    for c in underscore_cols:
        df = df.withColumn(c, regexp_replace(col(c), "_", ""))

    df = df.withColumn(
        "Amount_invested_monthly",
        F.when(col("Amount_invested_monthly") == "__10000__", None).otherwise(col("Amount_invested_monthly"))
    )
    df = df.withColumn(
        "Payment_Behaviour",
        F.when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour"))
    )
    df = df.withColumn(
        "Credit_Mix",
        F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix"))
    )
    df = df.withColumn(
        "Changed_Credit_Limit",
        F.when(col("Changed_Credit_Limit") == "", None).otherwise(col("Changed_Credit_Limit"))
    )
    df = df.withColumn(
        "Num_of_Loan",
        F.when(col("Num_of_Loan") == -100, None).otherwise(col("Num_of_Loan"))
    )
    df = df.withColumn(
        "Num_Bank_Accounts",
        F.when(col("Num_Bank_Accounts") == -1, None).otherwise(col("Num_Bank_Accounts"))
    )
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    #Interest rate should be less than 40, if not, set to null
    df = df.withColumn("Interest_Rate_clean", col("Interest_Rate").cast(FloatType()))
    df = df.withColumn("Interest_Rate_flag", col("Interest_Rate_clean") > 40)
    df = df.withColumn("Interest_Rate_clean", F.when(col("Interest_Rate_flag"), None).otherwise(col("Interest_Rate_clean")))

    # Num_Bank_Accounts: cast, flag negatives or > 15
    df = df.withColumn("Num_Bank_Accounts_clean", col("Num_Bank_Accounts").cast(IntegerType()))
    df = df.withColumn("Num_Bank_Accounts_flag",
        (col("Num_Bank_Accounts_clean") < 0) | (col("Num_Bank_Accounts_clean") > 15))
    df = df.withColumn("Num_Bank_Accounts_clean",
        F.when(col("Num_Bank_Accounts_flag"), None).otherwise(col("Num_Bank_Accounts_clean")))

    df = df.withColumn("Num_of_Loan_clean", col("Num_of_Loan").cast(IntegerType()))
    df = df.withColumn("Num_of_Loan_flag", col("Num_of_Loan_clean") >= 18)
    df = df.withColumn("Num_of_Loan_clean",
        F.when(col("Num_of_Loan_flag"), None).otherwise(col("Num_of_Loan_clean")))
    
    # Num_Credit_Card: flag > 12
    df = df.withColumn("Num_Credit_Card_clean", col("Num_Credit_Card").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Card_flag", col("Num_Credit_Card_clean") > 12)
    df = df.withColumn("Num_Credit_Card_clean",
        F.when(col("Num_Credit_Card_flag"), None).otherwise(col("Num_Credit_Card_clean")))

    # Num_Credit_Inquiries: cast, flag > 20
    df = df.withColumn("Num_Credit_Inquiries_clean", col("Num_Credit_Inquiries").cast(IntegerType()))
    df = df.withColumn("Num_Credit_Inquiries_flag", col("Num_Credit_Inquiries_clean") > 20)
    df = df.withColumn("Num_Credit_Inquiries_clean",
        F.when(col("Num_Credit_Inquiries_flag"), None).otherwise(col("Num_Credit_Inquiries_clean")))

    df = df.withColumn("Num_of_Delayed_Payment_clean", col("Num_of_Delayed_Payment").cast(IntegerType()))
    df = df.withColumn("Num_of_Delayed_Payment_flag",(col("Num_of_Delayed_Payment_clean") < 0) )
    df = df.withColumn("Num_of_Delayed_Payment_clean",
        F.when(col("Num_of_Delayed_Payment_flag"), None).otherwise(col("Num_of_Delayed_Payment_clean")))

    #create column credit history age in months, from string column Credit_History_Age
    parts = split(col("Credit_History_Age"), " ")
    df = df.withColumn("Credit_History_Age_in_mths", parts[0].cast(IntegerType()) * 12 + parts[3].cast(IntegerType()))

    delayed_rate = col("Num_of_Delayed_Payment_clean") / col("Credit_History_Age_in_mths")
    df = df.withColumn("Num_of_Delayed_Payment_flag",
        col("Num_of_Delayed_Payment_flag") | (delayed_rate > 1))
    df = df.withColumn("Num_of_Delayed_Payment_clean",
        F.when(delayed_rate > 1, None).otherwise(col("Num_of_Delayed_Payment_clean")))
    df = df.withColumn("delayed_payment_rate", delayed_rate)

    #tag accounts where ratio of EMI to monthly salary is greater than 1, and set EMI to null for those accounts
    df = df.withColumn("Total_EMI_per_month_clean", col("Total_EMI_per_month").cast(FloatType()))
    df = df.withColumn("Total_EMI_per_month_flag",
       F.coalesce((col("Total_EMI_per_month_clean") / col("Monthly_Inhand_Salary")) > 1, F.lit(False)))
    df = df.withColumn("Total_EMI_per_month_clean",
        F.when(col("Total_EMI_per_month_flag"), None).otherwise(col("Total_EMI_per_month_clean")))

    df = df.withColumn("Monthly_Balance_clean", col("Monthly_Balance").cast(FloatType()))
    df = df.withColumn("Monthly_Balance_clean",
        F.when(col("Monthly_Balance_clean") < 0, None).otherwise(col("Monthly_Balance_clean")))

    # Payment_Behaviour: split into spent_level and value_tier
    df = df.withColumn("Payment_Behaviour_clean", col("Payment_Behaviour").cast(StringType()))
    behaviour_parts = split(col("Payment_Behaviour_clean"), "_")
    df = df.withColumn("spent_level", behaviour_parts[0])
    df = df.withColumn("value_tier", behaviour_parts[2])

    df = df.withColumn("ingestion_timestamp", F.current_timestamp())

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df