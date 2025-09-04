# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ============== CONFIG ==============
SPLIT_DIR   = "/home/master/wzheng/projects/model_training/data/topN_splits"
TOPN_JSON   = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
MODEL_DIR   = "/home/master/wzheng/projects/model_training/models/v5_xgb"
REPORT_DIR  = "/home/master/wzheng/projects/model_training/reports/v5_xgb"
ALL_TRAIN   = "/home/master/wzheng/projects/model_training/data/train.csv"
ALL_VAL     = "/home/master/wzheng/projects/model_training/data/val.csv"
N_TRIALS    = 50
RANDOM_SEED = 42

TARGET_COL  = "RemoteWallClockTime"
TIME_COL    = "SubmitTime"
BUCKET_COL  = "BucketLabel"  # may or may not exist in files; handled robustly

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ============== Helpers ==============

def _numeric_features(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> list:
    cols = [c for c in df.columns if c not in drop_cols]
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return num_cols


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)


def compute_bucket_report(y_true: pd.Series, y_pred: np.ndarray, buckets: pd.Series) -> pd.DataFrame:
    dfm = pd.DataFrame({"y": y_true.values, "p": y_pred, "b": buckets.values})
    rows = []
    for b in sorted(dfm["b"].dropna().unique()):
        sub = dfm[dfm["b"] == b]
        if sub.empty:
            continue
        yt, yp = sub["y"].values, sub["p"].values
        rmse = np.sqrt(mean_squared_error(yt, yp))
        mae  = mean_absolute_error(yt, yp)
        mape = _safe_mape(yt, yp)
        bias = float(np.mean(yp - yt))
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = np.nan
        rows.append({
            "Bucket": int(b),
            "Count": len(sub),
            "RMSE": rmse,
            "MAE": mae,
            "MAPE(%)": mape,
            "Bias": bias,
            "R2": r2,
        })
    return pd.DataFrame(rows).sort_values("Bucket") if rows else pd.DataFrame(columns=["Bucket","Count","RMSE","MAE","MAPE(%)","Bias","R2"])


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = _safe_mape(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias, "R2": r2}


# ============== Training per PID ==============
with open(TOPN_JSON, 'r') as f:
    top_programids = json.load(f)

summary_rows = []

for pid in top_programids:
    train_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
    val_path   = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")
    if not os.path.exists(train_path):
        print(f"[SKIP] PID {pid}: train file not found")
        continue

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame(columns=df_train.columns)

    # drop rows with NaN target
    df_train = df_train.dropna(subset=[TARGET_COL])
    df_val   = df_val.dropna(subset=[TARGET_COL])

    # If no validation, skip training to avoid early_stopping issues
    if len(df_val) == 0:
        print(f"[TRAIN-ONLY] PID {pid}: validation empty -> skip model training for this PID")
        summary_rows.append({
            "ProgramID_encoded": pid,
            "TrainSize": len(df_train),
            "ValSize": 0,
            "Status": "train_only_no_val"
        })
        continue

    feature_cols = _numeric_features(df_train, drop_cols=(TARGET_COL, TIME_COL, BUCKET_COL))
    if not feature_cols:
        print(f"[WARN] PID {pid}: No numeric features left after filtering. Skipping.")
        continue

    X_tr, y_tr = df_train[feature_cols], df_train[TARGET_COL]
    X_va, y_va = df_val[feature_cols],   df_val[TARGET_COL]

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 2000,
            "random_state": RANDOM_SEED,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
        }
        model = XGBRegressor(**params).set_params(early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_va)
        return float(np.sqrt(mean_squared_error(y_va, pred)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    best_params.update({
        "n_estimators": 2000,
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
    })

    model = XGBRegressor(**best_params).set_params(early_stopping_rounds=50)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    val_pred = model.predict(X_va)

    # Overall & bucket metrics
    ov = overall_metrics(y_va.values, val_pred)
    bucket_df = compute_bucket_report(y_va, val_pred, df_val.get(BUCKET_COL, pd.Series(index=df_val.index, dtype="Int64")).fillna(-1))
    bucket_path = os.path.join(REPORT_DIR, f"pid_{pid}_bucket_metrics.csv")
    bucket_df.to_csv(bucket_path, index=False)

    model_path = os.path.join(MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib")
    dump(model, model_path)

    row = {
        "ProgramID_encoded": pid,
        "TrainSize": len(df_train),
        "ValSize": len(df_val),
        "RMSE": ov["RMSE"],
        "MAE": ov["MAE"],
        "MAPE(%)": ov["MAPE(%)"],
        "Bias": ov["Bias"],
        "R2": ov["R2"],
        "BucketReport": os.path.basename(bucket_path),
        "ModelPath": model_path,
        "BestParams": json.dumps(best_params),
    }
    summary_rows.append(row)
    print(f"PID {pid} | RMSE: {ov['RMSE']:.4f} | MAE: {ov['MAE']:.4f} | R2: {ov['R2']:.4f}")

# ============== Fallback: Others ==============
print("\n>> Training fallback model for Others (XGB)...")
df_all_tr = pd.read_csv(ALL_TRAIN)
df_all_va = pd.read_csv(ALL_VAL)

with open(TOPN_JSON, 'r') as f:
    top_ids_set = set(json.load(f))

df_others_tr = df_all_tr[~df_all_tr['ProgramID_encoded'].isin(top_ids_set)].copy()
df_others_va = df_all_va[~df_all_va['ProgramID_encoded'].isin(top_ids_set)].copy()

# Guard
if len(df_others_va) > 0:
    feat_cols = _numeric_features(df_others_tr, drop_cols=(TARGET_COL, TIME_COL, BUCKET_COL))
    Xtr, ytr = df_others_tr[feat_cols], df_others_tr[TARGET_COL]
    Xva, yva = df_others_va[feat_cols], df_others_va[TARGET_COL]

    def obj_others(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 2000,
            "random_state": RANDOM_SEED,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
        }
        m = XGBRegressor(**params).set_params(early_stopping_rounds=50)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred = m.predict(Xva)
        return float(np.sqrt(mean_squared_error(yva, pred)))

    study_o = optuna.create_study(direction="minimize")
    study_o.optimize(obj_others, n_trials=N_TRIALS)
    best_o = study_o.best_params
    best_o.update({
        "n_estimators": 2000,
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
    })

    model_o = XGBRegressor(**best_o).set_params(early_stopping_rounds=50)
    model_o.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred_o = model_o.predict(Xva)

    ov_o = overall_metrics(yva.values, pred_o)
    bucket_o = compute_bucket_report(yva, pred_o, df_others_va.get(BUCKET_COL, pd.Series(index=df_others_va.index, dtype="Int64")).fillna(-1))
    bucket_o_path = os.path.join(REPORT_DIR, "others_bucket_metrics.csv")
    bucket_o.to_csv(bucket_o_path, index=False)

    model_o_path = os.path.join(MODEL_DIR, "xgb_model_others_optuna.joblib")
    dump(model_o, model_o_path)

    summary_rows.append({
        "ProgramID_encoded": "Others",
        "TrainSize": len(df_others_tr),
        "ValSize": len(df_others_va),
        "RMSE": ov_o["RMSE"],
        "MAE": ov_o["MAE"],
        "MAPE(%)": ov_o["MAPE(%)"],
        "Bias": ov_o["Bias"],
        "R2": ov_o["R2"],
        "BucketReport": os.path.basename(bucket_o_path),
        "ModelPath": model_o_path,
        "BestParams": json.dumps(best_o),
    })
    print(f"Others | RMSE: {ov_o['RMSE']:.4f} | MAE: {ov_o['MAE']:.4f} | R2: {ov_o['R2']:.4f}")
else:
    print("[SKIP] Others: validation set is empty.")

# ============== Save summary ==============
sum_df = pd.DataFrame(summary_rows)
sum_path = os.path.join(REPORT_DIR, "v5_xgb_evaluation_summary.csv")
sum_df.to_csv(sum_path, index=False)
print("\nAll XGB models trained. Summary ->", sum_path)