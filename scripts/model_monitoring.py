import argparse
import os
import pickle
import pprint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.functions import col
from sklearn.metrics import roc_auc_score

import utils.model_bank

# to call this script: python scripts/model_monitoring.py --snapshotdate "2024-12-01" --modelname "credit_model_2024_05_01_xgboost.pkl"

# PSI / CSI: < 0.1 stable, 0.1 - 0.25 watch, > 0.25 action
# Gini: alert when it falls more than 10% below the test Gini recorded at training
GINI_ALERT_RATIO = 0.9


def psi(expected, actual, bins=10):
    # bins are the deciles of the baseline (training) distribution
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-4, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def stability_status(value):
    return "stable" if value < 0.1 else "watch" if value < 0.25 else "action"


def monitor_month(month_pdf, period, baseline, csi_features, gini_alert):
    row = {"loan_month": str(month_pdf["snapshot_date"].iloc[0]), "period": period, "n_scored": len(month_pdf),
           "mean_pd": month_pdf["model_predictions"].mean()}

    row["psi_score"] = psi(baseline["scores"], month_pdf["model_predictions"].values)
    row["psi_status"] = stability_status(row["psi_score"])
    for f in csi_features:
        row["csi_" + f] = psi(baseline["features"][f].dropna().values, month_pdf[f].dropna().values)
    row["csi_max_status"] = stability_status(max(row["csi_" + f] for f in csi_features))

    labelled = month_pdf[month_pdf["label"].notna()]
    row["n_labelled"] = len(labelled)
    if len(labelled) > 0 and labelled["label"].nunique() == 2:
        row["bad_rate"] = labelled["label"].mean()
        row["gini"] = 2 * roc_auc_score(labelled["label"], labelled["model_predictions"]) - 1
        row["gini_status"] = "ok" if row["gini"] >= gini_alert else "alert"
    else:
        row["bad_rate"], row["gini"], row["gini_status"] = np.nan, np.nan, "labels not mature"
    return row


def plot_monitoring(monitoring_pdf, csi_features, gini_test, gini_alert, title, filepath):
    m = monitoring_pdf.sort_values("loan_month").reset_index(drop=True)
    x = np.arange(len(m))
    # month on every tick (yy-mm); the period name only under the first month of each period
    labels = [month[2:7] + ("\n" + period if i == 0 or period != m["period"][i - 1] else "")
              for i, (month, period) in enumerate(zip(m["loan_month"], m["period"]))]
    # divider wherever the period changes (OOT | production)
    boundaries = [i - 0.5 for i in range(1, len(m)) if m["period"][i] != m["period"][i - 1]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axes[0]
    ax.plot(x, m["psi_score"], marker="o", linewidth=2, label="PSI (model score)")
    for f in csi_features:
        ax.plot(x, m["csi_" + f], marker="o", linestyle="--", label="CSI " + f)
    ax.axhline(0.1, color="orange", linestyle=":", label="watch (0.1)")
    ax.axhline(0.25, color="red", linestyle=":", label="action (0.25)")
    ax.set_title("Stability vs training baseline")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, m["gini"], marker="o", linewidth=2, label="Gini")
    ax.axhline(gini_test, color="green", linestyle="--", label=f"test Gini at training ({gini_test:.3f})")
    ax.axhline(gini_alert, color="red", linestyle=":", label=f"alert ({gini_alert:.3f})")
    ax.set_ylim(0, 1)
    ax.set_title("Performance (labels mature 6 months after loan start)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(x, m["mean_pd"], marker="o", label="predicted default rate")
    ax.plot(x, m["bad_rate"], marker="o", label="actual default rate")
    ax.set_ylim(0, 0.6)
    ax.set_title("Predicted vs actual default rate")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("loan month")
        for b in boundaries:
            ax.axvline(b, color="grey", linewidth=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["model_name"] = modelname
    config["model_artefact_filepath"] = "model_bank/" + modelname
    config["predictions_directory"] = "datamart/gold/model_predictions/" + modelname[:-4] + "/"
    config["monitoring_directory"] = "datamart/gold/model_monitoring/" + modelname[:-4] + "/"
    pprint.pprint(config)

    if not os.path.exists(config["model_artefact_filepath"]):
        print("skip: model not trained yet")
        return

    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    baseline = model_artefact["baseline"]
    csi_features = model_artefact["csi_features"]
    gini_test = model_artefact["results"]["gini_test"]
    gini_alert = GINI_ALERT_RATIO * gini_test
    role = utils.model_bank.get_role(modelname)   # as recorded in model_bank/champion.json at this run

    # --- OOT months: scored at training, labels already known ---
    month_frames = [("OOT", month_pdf) for _, month_pdf in model_artefact["oot_predictions"].groupby("snapshot_date")]

    # --- production months: predictions made so far, with labels known as of this run ---
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    if os.path.exists(config["predictions_directory"]):
        preds_pdf = spark.read.parquet(config["predictions_directory"] + "*.parquet") \
            .filter(col("snapshot_date") <= snapshotdate).toPandas()
        features_pdf = spark.read.parquet("datamart/gold/feature_store/gold_feature_store_*.parquet") \
            .select("loan_id", *csi_features).toPandas()
        # label snapshot = loan start + 6 months, so only labels observed by this run are used
        labels_pdf = spark.read.parquet("datamart/gold/label_store/gold_label_store_*.parquet") \
            .filter(col("snapshot_date") <= snapshotdate).select("loan_id", "label").toPandas()
        preds_pdf = preds_pdf.merge(features_pdf, on="loan_id", how="left").merge(labels_pdf, on="loan_id", how="left")
        # period: production (scored by the deployed model); older files without the column count as production
        if "period" not in preds_pdf.columns:
            preds_pdf["period"] = "production"
        month_frames += [(month_pdf["period"].iloc[0], month_pdf) for _, month_pdf in preds_pdf.groupby("snapshot_date")]

    # --- one monitoring row per loan month ---
    monitoring_pdf = pd.DataFrame([monitor_month(month_pdf, period, baseline, csi_features, gini_alert)
                                   for period, month_pdf in month_frames])
    monitoring_pdf.insert(0, "monitoring_date", snapshotdate)
    monitoring_pdf.insert(1, "model_name", modelname[:-4])
    monitoring_pdf.insert(2, "role", role)
    print(f"{role} {modelname} | test Gini at training: {gini_test} | alert below {gini_alert:.3f}")
    print(monitoring_pdf.drop(columns=["monitoring_date", "model_name"]).round(4).to_string(index=False))

    # --- save monitoring table to datamart gold ---
    os.makedirs(config["monitoring_directory"], exist_ok=True)
    date_suffix = snapshotdate.replace('-', '_')
    filepath = config["monitoring_directory"] + "monitoring_" + date_suffix + ".parquet"
    spark.createDataFrame(monitoring_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    # --- visualise performance and stability across time ---
    chart_filepath = config["monitoring_directory"] + "monitoring_" + date_suffix + ".png"
    plot_monitoring(monitoring_pdf, csi_features, gini_test, gini_alert,
                    f"{modelname[:-4]} ({role}) - monitoring as of {snapshotdate}", chart_filepath)
    print('saved to:', chart_filepath)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    args = parser.parse_args()

    main(args.snapshotdate, args.modelname)
