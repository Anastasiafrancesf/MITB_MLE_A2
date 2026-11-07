import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyspark

# to call this script: python scripts/model_alert.py --snapshotdate "2024-12-01" --modelnames "credit_model_2024_05_01_xgboost.pkl,credit_model_2024_05_01_logistic_regression.pkl"

# a breach = PSI / CSI in the "action" band (> 0.25) or Gini below the alert line (90% of test Gini)
# champion breach -> this task fails (red in Airflow) so someone investigates; challenger breach -> warning only


def plot_comparison(monitoring_pdf, title, filepath):
    months = sorted(monitoring_pdf["loan_month"].unique())
    x = np.arange(len(months))
    periods = [monitoring_pdf.loc[monitoring_pdf.loan_month == m, "period"].iloc[0] for m in months]
    # month on every tick (yy-mm); the period name only under the first month of each period
    labels = [m[2:7] + ("\n" + p if i == 0 or p != periods[i - 1] else "") for i, (m, p) in enumerate(zip(months, periods))]
    boundaries = [i - 0.5 for i in range(1, len(months)) if periods[i] != periods[i - 1]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (role, model_name), m in monitoring_pdf.groupby(["role", "model_name"]):
        m = m.set_index("loan_month").reindex(months)
        name = f"{role}: {model_name.split('_', 5)[-1]}"
        axes[0].plot(x, m["gini"], marker="o", linewidth=2, label=name)
        axes[1].plot(x, m["psi_score"], marker="o", linewidth=2, label=name)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Gini by loan month (labels mature 6 months after loan start)")
    axes[1].axhline(0.1, color="orange", linestyle=":", label="watch (0.1)")
    axes[1].axhline(0.25, color="red", linestyle=":", label="action (0.25)")
    axes[1].set_title("PSI of model score vs training baseline")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("loan month")
        ax.legend(fontsize=8)
        for b in boundaries:
            ax.axvline(b, color="grey", linewidth=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def main(snapshotdate, modelnames):
    print('\n\n---starting job---\n\n')
    date_suffix = snapshotdate.replace('-', '_')
    paths = [f"datamart/gold/model_monitoring/{m[:-4]}/monitoring_{date_suffix}.parquet" for m in modelnames]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("skip: no monitoring results yet")
        return

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    monitoring_pdf = spark.read.parquet(*paths).toPandas()
    spark.stop()

    # --- champion vs challenger chart ---
    chart_filepath = f"datamart/gold/model_monitoring/champion_vs_challenger_{date_suffix}.png"
    plot_comparison(monitoring_pdf, f"Champion vs challenger - monitoring as of {snapshotdate}", chart_filepath)
    print('saved to:', chart_filepath)

    # --- threshold check ---
    breach = (monitoring_pdf["psi_status"] == "action") | (monitoring_pdf["csi_max_status"] == "action") \
        | (monitoring_pdf["gini_status"] == "alert")
    breaches = monitoring_pdf.loc[breach, ["role", "model_name", "loan_month", "psi_score", "psi_status",
                                           "csi_max_status", "gini", "gini_status"]]
    if breaches.empty:
        print("no thresholds breached")
    else:
        print("THRESHOLD BREACHES:")
        print(breaches.round(4).to_string(index=False))
    if (breaches["role"] == "challenger").any():
        print("WARNING: challenger breached a threshold (no action on production scoring)")

    print('\n\n---completed job---\n\n')
    if (breaches["role"] == "champion").any():
        sys.exit("ALERT: champion model breached a monitoring threshold - investigate before the next run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelnames", type=str, required=True, help="comma-separated model file names")
    args = parser.parse_args()

    main(args.snapshotdate, args.modelnames.split(","))
