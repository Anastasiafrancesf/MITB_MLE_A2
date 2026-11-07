import argparse
import os
import pickle
import pprint
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyspark
import xgboost as xgb
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import col
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import utils.model_bank

# to call this script: python scripts/model_train.py --snapshotdate "2024-05-01" --modeltraindate "2024-05-01"

CAT_COLS = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "spent_level", "value_tier"]
NON_FEATURE_COLS = ["loan_id", "Customer_ID", "loan_start_date", "snapshot_date", "attr_date", "fin_date",
                    "tenure", "loan_amt",   # identical for every loan
                    "Type_of_Loan",         # raw text, represented by Type_of_Loan_count
                    "label"]
N_CSI_FEATURES = 3   # importance more than halves after the third feature
SELECTION_TOLERANCE = 0.02   # settings within this of the best validation Gini count as near-tied


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def smallest_gap_index(cv_results):
    # selection rule: among settings within SELECTION_TOLERANCE of the best validation Gini,
    # pick the one with the smallest train-validation gap (least overfitting)
    gini_val = 2 * cv_results["mean_test_score"] - 1
    gini_train = 2 * cv_results["mean_train_score"] - 1
    near_best = gini_val >= gini_val.max() - SELECTION_TOLERANCE
    return int(np.argmin(np.where(near_best, gini_train - gini_val, np.inf)))


def main(snapshotdate, modeltraindate):
    print('\n\n---starting job---\n\n')

    # the model is (re)trained on the scheduled training date only
    if snapshotdate != modeltraindate:
        print(f"skip: model is trained on {modeltraindate}")
        return

    # --- set up config (lab 5 structure: OOT = the months just before training, all labels already observed) ---
    config = {}
    config["model_train_date_str"] = snapshotdate
    # labels exist from 2023-07 (loans from 2023-01 + 6 months), so a 2024-05 training date leaves 8 months
    config["train_test_period_months"] = 8
    config["oot_period_months"] = 2
    config["train_test_ratio"] = 0.8
    config["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=config["oot_period_months"])
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=config["train_test_period_months"])
    pprint.pprint(config)

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- label store: label snapshot_date is the date the label was observed (loan start + 6 months) ---
    labels_sdf = spark.read.parquet("datamart/gold/label_store/gold_label_store_*.parquet") \
        .filter(col("snapshot_date").between(config["train_test_start_date"], config["oot_end_date"])) \
        .select("loan_id", "label", col("snapshot_date").alias("label_date"))

    # --- feature store: features as at loan start; joined on loan_id (dates differ by 6 months) ---
    features_sdf = spark.read.parquet("datamart/gold/feature_store/gold_feature_store_*.parquet")
    data_pdf = labels_sdf.join(features_sdf, on="loan_id", how="inner").toPandas()
    spark.stop()

    # includes clickstream (6-month averages); the model must go live while applicants are still in that feed
    num_cols = [c for c in data_pdf.columns if c not in NON_FEATURE_COLS + CAT_COLS + ["label_date"]]
    feature_cols = num_cols + CAT_COLS

    # --- split: train / test (random, same period) and OOT (later period) ---
    oot_pdf = data_pdf[data_pdf["label_date"] >= config["oot_start_date"].date()]
    train_test_pdf = data_pdf[data_pdf["label_date"] <= config["train_test_end_date"].date()]

    X_oot, y_oot = oot_pdf[feature_cols], oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"],
        test_size=1 - config["train_test_ratio"], random_state=88, shuffle=True, stratify=train_test_pdf["label"])

    data_stats = {name: {"rows": len(y), "bad_rate": round(float(y.mean()), 4)}
                  for name, y in [("train", y_train), ("test", y_test), ("oot", y_oot)]}
    pprint.pprint(data_stats)

    # --- candidate models; validation = 3-fold CV on train (hyperparameter tuning) ---
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=88)

    # logistic regression cannot take nulls: median fill + 0/1 "was missing" flag, learned on train only
    lr_pipeline = Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ])),
        ("model", LogisticRegression(solver="liblinear", max_iter=1000, random_state=88)),
    ])
    lr_search = GridSearchCV(lr_pipeline, {
        "model__C": [0.001, 0.01, 0.1, 1, 10],
        "model__penalty": ["l1", "l2"],
        "model__class_weight": [None, "balanced"],
    }, scoring="roc_auc", cv=cv, return_train_score=True, refit=smallest_gap_index, n_jobs=-1)

    # xgboost handles nulls natively
    xgb_pipeline = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ])),
        ("model", xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=88)),
    ])
    xgb_search = RandomizedSearchCV(xgb_pipeline, {
        "model__n_estimators": [100, 200, 400, 800],
        "model__max_depth": [1, 2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__min_child_weight": [5, 10, 20, 40],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.5, 0.7, 1.0],
        "model__gamma": [0, 0.5, 1, 2],
        "model__reg_lambda": [1, 5, 10, 20],
        "model__reg_alpha": [0, 1, 5],
    }, n_iter=50, scoring="roc_auc", cv=cv, return_train_score=True, refit=smallest_gap_index,
        random_state=42, n_jobs=-1)

    # --- train, then evaluate on train / validation / test / OOT ---
    comparison = {}
    searches = {"logistic_regression": lr_search, "xgboost": xgb_search}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        model = search.best_estimator_
        comparison[name] = {
            "gini_train": round(gini(y_train, model.predict_proba(X_train)[:, 1]), 4),
            "gini_validation": round(2 * search.cv_results_["mean_test_score"][search.best_index_] - 1, 4),
            "gini_test": round(gini(y_test, model.predict_proba(X_test)[:, 1]), 4),
            "gini_oot": round(gini(y_oot, model.predict_proba(X_oot)[:, 1]), 4),
        }
    print(pd.DataFrame(comparison).T.to_string())

    # --- recommended champion = best on test (validation only tunes each model); the other model runs alongside as challenger ---
    recommended = max(comparison, key=lambda n: comparison[n]["gini_test"])
    print("recommended champion:", recommended)

    os.makedirs("model_bank", exist_ok=True)
    saved = {}
    for model_type, search in searches.items():
        model = search.best_estimator_
        print(f"{model_type} tuned hyperparameters:", search.best_params_)

        # top features (permutation importance on test) are the ones monitored with CSI
        imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=88, n_jobs=-1)
        importance = pd.Series(imp.importances_mean, index=feature_cols).sort_values(ascending=False)
        print(f"{model_type} feature importance (drop in test AUC when shuffled):")
        print(importance.head(10).round(4).to_string())
        csi_features = [f for f in importance.index if f in num_cols][:N_CSI_FEATURES]

        # --- prepare model artefact to save ---
        model_artefact = {}
        model_artefact["model"] = model   # preprocessing (fill, flags, scaling, one-hot) + model in one pipeline
        model_artefact["model_version"] = "credit_model_" + snapshotdate.replace('-', '_') + "_" + model_type
        model_artefact["model_type"] = model_type
        model_artefact["feature_cols"] = feature_cols
        model_artefact["data_dates"] = config
        model_artefact["data_stats"] = data_stats
        model_artefact["results"] = comparison[model_type]
        model_artefact["model_comparison"] = comparison
        model_artefact["hp_params"] = search.best_params_
        model_artefact["feature_importance"] = importance.round(5).to_dict()
        model_artefact["csi_features"] = csi_features
        # monitoring baseline: training score and feature distributions (PSI / CSI compare against these)
        model_artefact["baseline"] = {
            "scores": model.predict_proba(X_train)[:, 1],
            "features": X_train[csi_features].reset_index(drop=True),
        }
        # OOT months are the first monitored months (their labels are already known at training)
        oot_monitoring_pdf = oot_pdf[["loan_id", "snapshot_date", "label"] + csi_features].copy()
        oot_monitoring_pdf["model_predictions"] = model.predict_proba(X_oot)[:, 1]
        model_artefact["oot_predictions"] = oot_monitoring_pdf.reset_index(drop=True)

        # --- save artefact to model bank ---
        file_path = os.path.join("model_bank", model_artefact["model_version"] + ".pkl")
        with open(file_path, "wb") as file:
            pickle.dump(model_artefact, file)
        saved[model_type] = os.path.basename(file_path)
        print("saved to", file_path)

    # --- deployment: model_bank/champion.json points at the production model ---
    # first deployment sets the pointer; after that, new models wait as candidates until an approved
    # edit of champion.json promotes one (human review before operationalising)
    if utils.model_bank.read_champion() is None:
        pointer = utils.model_bank.write_champion(
            saved[recommended], [m for t, m in saved.items() if t != recommended],
            reason=f"first deployment: best test Gini ({comparison[recommended]['gini_test']})")
        print("champion.json created:", pointer)
    else:
        print("champion.json unchanged:", utils.model_bank.read_champion()["champion"],
              "| new candidates saved; promote by editing champion.json after review")

    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modeltraindate", type=str, required=True, help="YYYY-MM-DD the model is trained on")
    args = parser.parse_args()

    main(args.snapshotdate, args.modeltraindate)
