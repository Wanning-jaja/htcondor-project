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

import lightgbm as lgb
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ============== CONFIG ==============
SPLIT_DIR   = "/home/master/wzheng/projects/model_training/data/topN_splits"
TOPN_JSON   = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
MODEL_DIR   = "/home/master/wzheng/projects/model_training/models/v5.1_lgb"
REPORT_DIR  = "/home/master/wzheng/projects/model_training/reports/v5.1_lgb"
ALL_TRAIN   = "/home/master/wzheng/projects/model_training/data/train.csv"
ALL_VAL     = "/home/master/wzheng/projects/model_training/data/val.csv"
N_TRIALS    = 50
RANDOM_SEED = 42

TARGET_COL  = "RemoteWallClockTime"
TIME_COL    = "SubmitTime"
BUCKET_COL  = "BucketLabel"  # 可能不存在

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

# ====== 与切分脚本一致的分桶边界 ======
# 0:[0,600) 1:[600,1800) 2:[1800,3600) 3:[3600,7200) 4:[7200,14400)
# 5:[14400,21600) 6:[21600,28800) 7:[28800,43200) 8:[43200,86400) 9:[86400,+inf)
_BUCKET_EDGES = [600, 1800, 3600, 7200, 14400, 21600, 28800, 43200, 86400]

def _bucketize_seconds(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return np.digitize(arr, _BUCKET_EDGES, right=False)  # 0..9

def _overall_bucket_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, true_buckets: pd.Series | None
) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    tb = true_buckets.to_numpy() if (true_buckets is not None and true_buckets.notna().any()) else _bucketize_seconds(yt)
    pb = _bucketize_seconds(yp)
    acc_strict = float(np.mean(pb == tb)) if len(tb) else np.nan
    acc_relax  = float(np.mean(np.abs(pb - tb) <= 1)) if len(tb) else np.nan
    return {"Accuracy": acc_strict, "Accuracy+-1": acc_relax}

def compute_bucket_report(y_true: pd.Series,
                          y_pred: np.ndarray,
                          true_buckets: pd.Series | None) -> pd.DataFrame:
    yt = y_true.to_numpy(dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    tb = true_buckets.to_numpy() if (true_buckets is not None and true_buckets.notna().any()) else _bucketize_seconds(yt)
    pb = _bucketize_seconds(yp)

    dfm = pd.DataFrame({"y": yt, "p": yp, "tb": tb, "pb": pb})
    rows = []
    for b in sorted(pd.unique(dfm["tb"])):
        sub = dfm[dfm["tb"] == b]
        if sub.empty:
            continue
        yt_b, yp_b = sub["y"].values, sub["p"].values
        rmse = float(np.sqrt(mean_squared_error(yt_b, yp_b)))
        mae  = float(mean_absolute_error(yt_b, yp_b))
        mape = float(np.mean(np.abs((yp_b - yt_b) / np.clip(np.abs(yt_b), 1e-9, None))) * 100.0)
        bias = float(np.mean(yp_b - yt_b))
        try:
            r2 = float(r2_score(yt_b, yp_b))
        except Exception:
            r2 = np.nan

        acc_strict = float(np.mean(sub["pb"].values == sub["tb"].values))
        acc_relax  = float(np.mean(np.abs(sub["pb"].values - sub["tb"].values) <= 1))

        rows.append({
            "Bucket": int(b),
            "Count": int(len(sub)),
            "RMSE": rmse,
            "MAE": mae,
            "MAPE(%)": mape,
            "Bias": bias,
            "R2": r2,
            "Accuracy": acc_strict,
            "Accuracy+-1": acc_relax,
        })

    cols = ["Bucket","Count","RMSE","MAE","MAPE(%)","Bias","R2","Accuracy","Accuracy+-1"]
    return pd.DataFrame(rows, columns=cols).sort_values("Bucket") if rows else pd.DataFrame(columns=cols)

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

    df_train = df_train.dropna(subset=[TARGET_COL])
    df_val   = df_val.dropna(subset=[TARGET_COL])

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
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "n_estimators": 3000,
            "random_state": RANDOM_SEED,
        }
        model = LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_va, num_iteration=model.best_iteration_)
        return float(np.sqrt(mean_squared_error(y_va, pred)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    best_params.update({
        "n_estimators": 3000,
        "random_state": RANDOM_SEED,
    })

    model = LGBMRegressor(**best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    val_pred = model.predict(X_va, num_iteration=model.best_iteration_)

    ov = overall_metrics(y_va.values, val_pred)
    accs = _overall_bucket_accuracy(y_va.values, val_pred, df_val.get(BUCKET_COL))
    bucket_df = compute_bucket_report(y_va, val_pred, df_val.get(BUCKET_COL))
    bucket_path = os.path.join(REPORT_DIR, f"pid_{pid}_bucket_metrics.csv")
    bucket_df.to_csv(bucket_path, index=False)

    model_path = os.path.join(MODEL_DIR, f"lgb_model_pid{pid}_optuna.joblib")
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
        "Accuracy": accs["Accuracy"],
        "Accuracy+-1": accs["Accuracy+-1"],
        "BucketReport": os.path.basename(bucket_path),
        "ModelPath": model_path,
        "BestParams": json.dumps(best_params),
    }
    summary_rows.append(row)
    print(f"PID {pid} | RMSE: {ov['RMSE']:.4f} | MAE: {ov['MAE']:.4f} | R2: {ov['R2']:.4f} | Acc: {accs['Accuracy']:.3f} | Acc+-1: {accs['Accuracy+-1']:.3f}")

# ============== Fallback: Others ==============
print("\n>> Training fallback model for Others (LightGBM)...")
df_all_tr = pd.read_csv(ALL_TRAIN)
df_all_va = pd.read_csv(ALL_VAL)

with open(TOPN_JSON, 'r') as f:
    top_ids_set = set(json.load(f))

df_others_tr = df_all_tr[~df_all_tr['ProgramID_encoded'].isin(top_ids_set)].copy()
df_others_va = df_all_va[~df_all_va['ProgramID_encoded'].isin(top_ids_set)].copy()

if len(df_others_va) > 0:
    feat_cols = _numeric_features(df_others_tr, drop_cols=(TARGET_COL, TIME_COL, BUCKET_COL))
    Xtr, ytr = df_others_tr[feat_cols], df_others_tr[TARGET_COL]
    Xva, yva = df_others_va[feat_cols], df_others_va[TARGET_COL]

    def obj_others(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 512),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "n_estimators": 3000,
            "random_state": RANDOM_SEED,
        }
        m = LGBMRegressor(**params)
        m.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        pred = m.predict(Xva, num_iteration=m.best_iteration_)
        return float(np.sqrt(mean_squared_error(yva, pred)))

    study_o = optuna.create_study(direction="minimize")
    study_o.optimize(obj_others, n_trials=N_TRIALS)
    best_o = study_o.best_params
    best_o.update({
        "n_estimators": 3000,
        "random_state": RANDOM_SEED,
    })

    model_o = LGBMRegressor(**best_o)
    model_o.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    pred_o = model_o.predict(Xva, num_iteration=model_o.best_iteration_)

    ov_o = overall_metrics(yva.values, pred_o)
    accs_o = _overall_bucket_accuracy(yva.values, pred_o, df_others_va.get(BUCKET_COL))
    bucket_o = compute_bucket_report(yva, pred_o, df_others_va.get(BUCKET_COL))
    bucket_o_path = os.path.join(REPORT_DIR, "others_bucket_metrics.csv")
    bucket_o.to_csv(bucket_o_path, index=False)

    model_o_path = os.path.join(MODEL_DIR, "lgb_model_others_optuna.joblib")
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
        "Accuracy": accs_o["Accuracy"],
        "Accuracy+-1": accs_o["Accuracy+-1"],
        "BucketReport": os.path.basename(bucket_o_path),
        "ModelPath": model_o_path,
        "BestParams": json.dumps(best_o),
    })
    print(f"Others | RMSE: {ov_o['RMSE']:.4f} | MAE: {ov_o['MAE']:.4f} | R2: {ov_o['R2']:.4f} | Acc: {accs_o['Accuracy']:.3f} | Acc+-1: {accs_o['Accuracy+-1']:.3f}")
else:
    print("[SKIP] Others: validation set is empty.")

# ============== Save summary ==============
sum_df = pd.DataFrame(summary_rows)
sum_path = os.path.join(REPORT_DIR, "v5.1_lgb_evaluation_summary.csv")
sum_df.to_csv(sum_path, index=False)
print("\nAll LightGBM models trained. Summary ->", sum_path)
