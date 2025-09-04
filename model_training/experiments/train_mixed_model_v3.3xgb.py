# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, warnings
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ====== 路径（与集成脚本一致） ======
SPLIT_DIR    = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON    = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"
MODEL_DIR    = "/home/master/wzheng/projects/model_training/models/v3.3_xgb"
REPORT_DIR   = "/home/master/wzheng/projects/model_training/reports/v3.3_xgb_reg"
ALL_TRAIN    = "/home/master/wzheng/projects/model_training/data/40train.csv"
ALL_VAL      = "/home/master/wzheng/projects/model_training/data/40val.csv"
FEATURES_JSON= "/home/master/wzheng/projects/model_training/models/v3.3_features.json"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TARGET_COL = "RemoteWallClockTime"
TIME_COL   = "SubmitTime"
BUCKET_COL = "BucketLabel"  # 可能不存在，但如果有要排除

N_TRIALS    = 50
RANDOM_SEED = 42

def _numeric_features(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> list:
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].select_dtypes(include=[np.number]).columns.tolist()

def _maybe_save_features(cols: list):
    try:
        if not os.path.exists(FEATURES_JSON):
            with open(FEATURES_JSON, "w", encoding="utf-8") as f:
                json.dump({"feature_cols": cols}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[WARN] save features.json failed:", e)

def make_internal_holdout(df: pd.DataFrame, time_col: str, target_col: str, feature_cols: list,
                          holdout_ratio: float = 0.05, min_holdout: int = 200, max_holdout: int = 20000):
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    n_ho = max(min(int(n*holdout_ratio), max_holdout), min_holdout)
    if n <= n_ho:
        return df_sorted[feature_cols], df_sorted[target_col], None, None
    df_tr2 = df_sorted.iloc[:n-n_ho]; df_ho = df_sorted.iloc[n-n_ho:]
    return df_tr2[feature_cols], df_tr2[target_col], df_ho[feature_cols], df_ho[target_col]

def overall_metrics(y_true, y_pred) -> Dict[str,float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    try: r2 = r2_score(y_true, y_pred)
    except: r2 = np.nan
    return {"RMSE":rmse,"MAE":mae,"Bias":bias,"R2":r2}

# ====== TopN per-PID ======
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

    # 仅数值特征，排除时间/目标/桶
    drop_cols = (TARGET_COL, TIME_COL, BUCKET_COL) if BUCKET_COL in df_train.columns else (TARGET_COL, TIME_COL)
    feature_cols = _numeric_features(df_train, drop_cols=drop_cols)
    if not feature_cols:
        print(f"[WARN] PID {pid}: No numeric features left after filtering. Skipping.")
        continue
    _maybe_save_features(feature_cols)

    X_tr_full, y_tr_full = df_train[feature_cols], df_train[TARGET_COL]

    # 验证选择：优先 val，否则内部尾部留出
    if len(df_val) > 0:
        X_eval, y_eval = df_val[feature_cols], df_val[TARGET_COL]
        eval_mode = "val"
    else:
        X_tr2, y_tr2, X_ho, y_ho = make_internal_holdout(df_train, TIME_COL, TARGET_COL, feature_cols)
        if X_ho is not None and len(X_ho)>0:
            X_eval, y_eval = X_ho, y_ho
            X_tr_full, y_tr_full = X_tr2, y_tr2
            eval_mode = "holdout"
        else:
            X_eval, y_eval = None, None
            eval_mode = "train_only"

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
        model = XGBRegressor(**params)
        if X_eval is not None:
            model.set_params(early_stopping_rounds=50)
            model.fit(X_tr_full, y_tr_full, eval_set=[(X_eval, y_eval)], verbose=False)
            pred = model.predict(X_eval)
            return float(np.sqrt(mean_squared_error(y_eval, pred)))
        else:
            model.set_params(n_estimators=800)
            model.fit(X_tr_full, y_tr_full, verbose=False)
            pred_tr = model.predict(X_tr_full)
            return float(np.sqrt(mean_squared_error(y_tr_full, pred_tr)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    best_params.update({
        "n_estimators": 2000 if X_eval is not None else 800,
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
    })

    final_model = XGBRegressor(**best_params)
    if X_eval is not None:
        final_model.set_params(early_stopping_rounds=50)
        final_model.fit(X_tr_full, y_tr_full, eval_set=[(X_eval, y_eval)], verbose=False)
        eval_pred = final_model.predict(X_eval)
        ov = overall_metrics(y_eval, eval_pred)
    else:
        final_model.fit(X_tr_full, y_tr_full, verbose=False)
        ov = {"RMSE":np.nan,"MAE":np.nan,"Bias":np.nan,"R2":np.nan}

    model_path = os.path.join(MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib")
    dump(final_model, model_path)

    summary_rows.append({
        "ProgramID_encoded": pid,
        "TrainSize": len(df_train),
        "ValSize": len(df_val),
        "EvalMode": "val" if X_eval is not None and len(df_val)>0 else ("holdout" if X_eval is not None else "train_only"),
        **ov,
        "ModelPath": model_path,
        "BestParams": json.dumps(best_params),
    })
    print(f"PID {pid} | RMSE:{ov['RMSE']:.4f} | MAE:{ov['MAE']:.4f} | R2:{ov['R2']:.4f} | mode={summary_rows[-1]['EvalMode']}")

# ====== Others（非 TopN）======
print("\n>> Training fallback model for Others (XGB)…")
df_all_tr = pd.read_csv(ALL_TRAIN)
df_all_va = pd.read_csv(ALL_VAL)
with open(TOPN_JSON, 'r') as f: top_ids_set = set(json.load(f))
df_others_tr = df_all_tr[~df_all_tr['ProgramID_encoded'].isin(top_ids_set)].copy()
df_others_va = df_all_va[~df_all_va['ProgramID_encoded'].isin(top_ids_set)].copy()

if len(df_others_tr) > 0:
    drop_cols = (TARGET_COL, TIME_COL, BUCKET_COL) if BUCKET_COL in df_others_tr.columns else (TARGET_COL, TIME_COL)
    feat_cols = _numeric_features(df_others_tr, drop_cols)
    _maybe_save_features(feat_cols)

    Xtr, ytr = df_others_tr[feat_cols], df_others_tr[TARGET_COL]

    if len(df_others_va) > 0:
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
                "n_estimators": 2000, "random_state": RANDOM_SEED,
                "objective": "reg:squarederror", "eval_metric": "rmse", "verbosity": 0,
            }
            m = XGBRegressor(**params).set_params(early_stopping_rounds=50)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pred = m.predict(Xva)
            return float(np.sqrt(mean_squared_error(yva, pred)))

        study_o = optuna.create_study(direction="minimize")
        study_o.optimize(obj_others, n_trials=N_TRIALS)
        best_o = study_o.best_params
        best_o.update({"n_estimators":2000,"random_state":RANDOM_SEED,"objective":"reg:squarederror","eval_metric":"rmse","verbosity":0})
        model_o = XGBRegressor(**best_o).set_params(early_stopping_rounds=50)
        model_o.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        pred_o = model_o.predict(Xva)
        rmse_o = float(np.sqrt(mean_squared_error(yva, pred_o)))
        mae_o  = float(mean_absolute_error(yva, pred_o))
    else:
        model_o = XGBRegressor(n_estimators=800, random_state=RANDOM_SEED,
                               objective="reg:squarederror", eval_metric="rmse", verbosity=0)
        model_o.fit(Xtr, ytr, verbose=False)
        rmse_o = np.nan; mae_o = np.nan

    model_o_path = os.path.join(MODEL_DIR, "xgb_model_others_optuna.joblib")
    dump(model_o, model_o_path)
    summary_rows.append({"ProgramID_encoded":"Others","TrainSize":len(df_others_tr),"ValSize":len(df_others_va),
                         "EvalMode":"val" if len(df_others_va)>0 else "train_only",
                         "RMSE":rmse_o,"MAE":mae_o,"Bias":np.nan,"R2":np.nan,
                         "ModelPath":model_o_path,"BestParams":json.dumps(best_o) if len(df_others_va)>0 else "{}"})
else:
    print("[SKIP] Others: no training data.")

# ====== 保存汇总 ======
sum_df = pd.DataFrame(summary_rows)
sum_df.to_csv(os.path.join(REPORT_DIR, "v3.3_xgb_reg_summary.csv"), index=False)
print("Saved summary to:", os.path.join(REPORT_DIR, "v3.3_xgb_reg_summary.csv"))
