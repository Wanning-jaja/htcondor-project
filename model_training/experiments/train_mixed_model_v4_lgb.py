# -*- coding: utf-8 -*-
from lightgbm import LGBMClassifier, early_stopping
import os
import json
import pandas as pd
import numpy as np
import optuna
from joblib import dump

# === 配置 ===
SPLIT_DIR = "/home/master/wzheng/projects/model_training/data/topN_splits"
TOPN_JSON = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v4.0_lgb"
REPORT_DIR = "/home/master/wzheng/projects/model_training/reports/v4.0_lgb"
ALL_TRAIN = "/home/master/wzheng/projects/model_training/data/train.csv"
ALL_VAL = "/home/master/wzheng/projects/model_training/data/val.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

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

# === 单独训练 TopN ===
for pid in top_programids:
    train_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
    val_path = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")
    if not os.path.exists(train_path): continue
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    df_train["BucketLabel"] = df_train["RemoteWallClockTime"].apply(get_bucket_label)
    df_val["BucketLabel"] = df_val["RemoteWallClockTime"].apply(get_bucket_label)
    valid_labels = set(df_train["BucketLabel"].unique())
    df_val = df_val[df_val["BucketLabel"].isin(valid_labels)]
    
    feature_cols = [c for c in df_train.columns if c not in ['RemoteWallClockTime', 'SubmitTime', 'BucketLabel']]
    target_col = 'BucketLabel'

    def objective(trial):
        try:
            param = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "n_estimators": 1000,
                "random_state": 42
            }
            model = LGBMClassifier(**param)
            model.fit(
                df_train[feature_cols], df_train[target_col],
                eval_set=[(df_val[feature_cols], df_val[target_col])],
                eval_metric="multi_logloss", callbacks=[early_stopping(10)]
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
    best_params.update({"n_estimators": 1000, "random_state": 42})
    model = LGBMClassifier(**best_params)
    model.fit(
        df_train[feature_cols], df_train[target_col],
        eval_set=[(df_val[feature_cols], df_val[target_col])],
        eval_metric="multi_logloss", callbacks=[early_stopping(10)]
    )
    val_pred = model.predict(df_val[feature_cols])
    acc = (val_pred == df_val[target_col]).mean()
    over = (val_pred > df_val[target_col]).sum()
    under = (val_pred < df_val[target_col]).sum()
    dev = np.abs(val_pred - df_val[target_col]).mean()

    dump(model, os.path.join(MODEL_DIR, f"lgb_model_pid{pid}_optuna.joblib"))
    results.append({
        "ProgramID_encoded": pid,
        "TrainSize": len(df_train),
        "ValSize": len(df_val),
        "Accuracy": acc,
        "OverPredicted": int(over),
        "UnderPredicted": int(under),
        "MeanDeviation": float(dev),
        "BestParams": json.dumps(best_params)
    })
    print(f"{pid} | Accuracy: {acc:.4f} | Over: {over} | Under: {under} | Dev: {dev:.3f}")

# === Others ===
print("\n>> Training fallback model for Others (LightGBM)...")
df_train = pd.read_csv(ALL_TRAIN)
df_val = pd.read_csv(ALL_VAL)
df_others_train = df_train[~df_train['ProgramID_encoded'].isin(top_programids)]
df_others_val = df_val[~df_val['ProgramID_encoded'].isin(top_programids)]
df_others_train["BucketLabel"] = df_others_train["RemoteWallClockTime"].apply(get_bucket_label)
df_others_val["BucketLabel"] = df_others_val["RemoteWallClockTime"].apply(get_bucket_label)
#valid_labels = set(df_train["BucketLabel"].unique())
#df_val = df_val[df_val["BucketLabel"].isin(valid_labels)]
valid_labels = set(df_others_train["BucketLabel"].unique())
df_others_val = df_others_val[df_others_val["BucketLabel"].isin(valid_labels)]

feature_cols = [c for c in df_others_train.columns if c not in ['RemoteWallClockTime', 'SubmitTime', 'BucketLabel']]
target_col = 'BucketLabel'

def objective_others(trial):
    try:
        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 1000,
            "random_state": 42
        }
        model = LGBMClassifier(**param)
        model.fit(
            df_others_train[feature_cols], df_others_train[target_col],
            eval_set=[(df_others_val[feature_cols], df_others_val[target_col])],
            eval_metric="multi_logloss", callbacks=[early_stopping(10)]
        )
        pred = model.predict(df_others_val[feature_cols])
        acc = (pred == df_others_val[target_col]).mean()
        return 1.0 - acc
    except Exception as e:
        print(f"Trial failed for Others: {e}")
        return float("inf")

study_o = optuna.create_study(direction="minimize")
study_o.optimize(objective_others, n_trials=50)

best_o = study_o.best_params
best_o.update({"n_estimators": 1000, "random_state": 42})
model_o = LGBMClassifier(**best_o)
model_o.fit(
    df_others_train[feature_cols], df_others_train[target_col],
    eval_set=[(df_others_val[feature_cols], df_others_val[target_col])],
    eval_metric="multi_logloss", callbacks=[early_stopping(10)]
)
pred_o = model_o.predict(df_others_val[feature_cols])
acc_o = (pred_o == df_others_val[target_col]).mean()
over_o = (pred_o > df_others_val[target_col]).sum()
under_o = (pred_o < df_others_val[target_col]).sum()
dev_o = np.abs(pred_o - df_others_val[target_col]).mean()
dump(model_o, os.path.join(MODEL_DIR, "lgb_model_others_optuna.joblib"))

results.append({
    "ProgramID_encoded": "Others",
    "TrainSize": len(df_others_train),
    "ValSize": len(df_others_val),
    "Accuracy": acc_o,
    "OverPredicted": int(over_o),
    "UnderPredicted": int(under_o),
    "MeanDeviation": float(dev_o),
    "BestParams": json.dumps(best_o)
})

report_path = os.path.join(REPORT_DIR, "v4.0_lgb_evaluation_summary.csv")
pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).to_csv(report_path, index=False)
print("\nAll LightGBM models trained and results saved to:", report_path)
