# shared by both pipelines (monthly and daily scoring)
# lab 5 structure: train on 2024-05-01 with train/test labels + 2 months OOT, all observed before that date.
# The model uses clickstream, so it must go live while new applicants are still in the clickstream feed (until 2024-06).
MODEL_TRAIN_DATE = "2024-05-01"
# both tuned models are saved: the champion (best test Gini) scores production, the challenger runs alongside
MODEL_TYPES = ["xgboost", "logistic_regression"]
MODEL_NAMES = {t: "credit_model_" + MODEL_TRAIN_DATE.replace("-", "_") + "_" + t + ".pkl" for t in MODEL_TYPES}
