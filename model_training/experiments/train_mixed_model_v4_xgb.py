# -*- coding: utf-8 -*-
from xgboost import XGBClassifier
import inspect

import os
import json
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
print("XGBoost version:", xgb.__version__)
print("XGBoost path:", xgb.__file__)
from joblib import dump

# === 配置 ===
SPLIT_DIR = "/home/master/wzheng/projects/model_training/data/topN_splits"
TOPN_JSON = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v4.0_xgb"
REPORT_DIR = "/home/master/wzheng/projects/model_training/reports/v4.0_xgb"
ALL_TRAIN = "/home/master/wzheng/projects/model_training/data/train.csv"
ALL_VAL = "/home/master/wzheng/projects/model_training/data/val.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# === 加载 TopN ProgramID 列表 ===
with open(TOPN_JSON, 'r') as f:
    top_programids = json.load(f)

results = []


def get_bucket_label(sec):
    if sec < 600: return 0
    elif sec < 1800: return 1
    elif sec < 3600: return 2
    elif sec < 7200: return 3
    elif sec < 14400: return 4
    elif sec < 21600: return 5
    elif sec < 28800: return 6
    elif sec < 43200: return 7
    elif sec < 86400: return 8
    else: return 9


def compute_directional_errors(true, pred):
    diff = pred - true
    over = np.sum(diff > 0)
    under = np.sum(diff < 0)
    mean_dev = np.abs(diff).mean()
    return over, under, mean_dev


# === 遍历每个 Top-N ProgramID 训练独立模型 ===
for pid in top_programids:
    train_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
    val_path = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        continue
    df_train = pd.read_csv(train_path, encoding='latin1')
    df_val = pd.read_csv(val_path, encoding='latin1')

    df_train["BucketLabel"] = df_train["RemoteWallClockTime"].apply(get_bucket_label)
    df_val["BucketLabel"] = df_val["RemoteWallClockTime"].apply(get_bucket_label)

    feature_cols = [c for c in df_train.columns if c not in ['RemoteWallClockTime', 'SubmitTime', 'BucketLabel']]
    target_col = 'BucketLabel'

    def objective(trial):
        try:
            param = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "n_estimators": 1000,
                "random_state": 42,
                "objective": "multi:softmax",
                "num_class": 10,
                "eval_metric": "mlogloss",
                "verbosity": 0
            }
            model = XGBClassifier(**param).set_params(early_stopping_rounds=10)
            model.fit(
                df_train[feature_cols],
                df_train[target_col],
                eval_set=[(df_val[feature_cols], df_val[target_col])],
                verbose=False
            )
            pred = model.predict(df_val[feature_cols])
            acc = (pred == df_val[target_col]).mean()
            return 1.0 - acc
        except Exception as e:
            print(f"Trial failed for pid={pid}: {e}")
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    best_params.update({
        "n_estimators": 1000,
        "random_state": 42,
        "objective": "multi:softmax",
        "num_class": 10,
        "eval_metric": "mlogloss",
        "verbosity": 0
    })
    model = XGBClassifier(**best_params).set_params(early_stopping_rounds=10)
    model.fit(
        df_train[feature_cols],
        df_train[target_col],
        eval_set=[(df_val[feature_cols], df_val[target_col])],
        verbose=False
    )
    val_pred = model.predict(df_val[feature_cols])
    acc = (val_pred == df_val[target_col]).mean()
    over, under, mean_dev = compute_directional_errors(df_val[target_col].values, val_pred)
    model_path = os.path.join(MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib")
    dump(model, model_path)
    results.append({
        "ProgramID_encoded": pid,
        "TrainSize": len(df_train),
        "ValSize": len(df_val),
        "Accuracy": acc,
        "OverPredicted": over,
        "UnderPredicted": under,
        "MeanDeviation": mean_dev,
        "BestParams": json.dumps(best_params)
    })
    print(f"{pid} | Accuracy: {acc:.4f} | Over: {over} | Under: {under} | MeanDev: {mean_dev:.2f}")

# === fallback model (Others) ===
print("\n>> Training fallback model for Others (Optuna)...")
df_all_train = pd.read_csv(ALL_TRAIN, encoding='latin1')
df_all_val = pd.read_csv(ALL_VAL, encoding='latin1')
df_others_train = df_all_train[~df_all_train['ProgramID_encoded'].isin(top_programids)]
df_others_val = df_all_val[~df_all_val['ProgramID_encoded'].isin(top_programids)]
df_others_train["BucketLabel"] = df_others_train["RemoteWallClockTime"].apply(get_bucket_label)
df_others_val["BucketLabel"] = df_others_val["RemoteWallClockTime"].apply(get_bucket_label)
feature_cols = [col for col in df_others_train.columns if col not in ['RemoteWallClockTime', 'SubmitTime', 'BucketLabel']]
target_col = 'BucketLabel'

study_others = optuna.create_study(direction="minimize")
def objective_others(trial):
    try:
        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 1000,
            "random_state": 42,
            "objective": "multi:softmax",
            "num_class": 10,
            "eval_metric": "mlogloss",
            "verbosity": 0
        }
        model = XGBClassifier(**param).set_params(early_stopping_rounds=10)
        model.fit(
            df_others_train[feature_cols],
            df_others_train[target_col],
            eval_set=[(df_others_val[feature_cols], df_others_val[target_col])],
            verbose=False
        )
        pred = model.predict(df_others_val[feature_cols])
        acc = (pred == df_others_val[target_col]).mean()
        return 1.0 - acc
    except Exception as e:
        print(f"Trial failed for Others: {e}")
        return float("inf")

study_others.optimize(objective_others, n_trials=50)
best_params = study_others.best_params
best_params.update({
    "n_estimators": 1000,
    "random_state": 42,
    "objective": "multi:softmax",
    "num_class": 10,
    "eval_metric": "mlogloss",
    "verbosity": 0
})
model_others = XGBClassifier(**best_params).set_params(early_stopping_rounds=10)
model_others.fit(
    df_others_train[feature_cols],
    df_others_train[target_col],
    eval_set=[(df_others_val[feature_cols], df_others_val[target_col])],
    verbose=False
)
val_pred_others = model_others.predict(df_others_val[feature_cols])
acc_others = (val_pred_others == df_others_val[target_col]).mean()
over, under, mean_dev = compute_directional_errors(df_others_val[target_col].values, val_pred_others)
model_path_others = os.path.join(MODEL_DIR, "xgb_model_others_optuna.joblib")
dump(model_others, model_path_others)

results.append({
    "ProgramID_encoded": "Others",
    "TrainSize": len(df_others_train),
    "ValSize": len(df_others_val),
    "Accuracy": acc_others,
    "OverPredicted": over,
    "UnderPredicted": under,
    "MeanDeviation": mean_dev,
    "BestParams": json.dumps(best_params)
})

print(f" Others | Accuracy: {acc_others:.4f} | Over: {over} | Under: {under} | MeanDev: {mean_dev:.2f}")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)
report_path = os.path.join(REPORT_DIR, "v4.0xgb_evaluation_summary.csv")
results_df.to_csv(report_path, index=False)
print("\n All model training is complete and the evaluation report is saved:", report_path)
