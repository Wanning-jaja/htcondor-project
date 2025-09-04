# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, warnings
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump

warnings.filterwarnings("ignore", category=UserWarning)

# ====== 路径 ======
SPLIT_DIR    = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON    = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"
MODEL_DIR    = "/home/master/wzheng/projects/test2/models/v3_90_lgb"
REPORT_DIR   = "/home/master/wzheng/projects/test2/reports/v3_90_lgb"
ALL_TRAIN    = "/home/master/wzheng/projects/model_training/data/40train.csv"
ALL_VAL      = "/home/master/wzheng/projects/model_training/data/40val.csv"
FEATURES_JSON= os.path.join(MODEL_DIR, "features.json")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ====== 任务列 ======
TARGET_COL = "RemoteWallClockTime"
TIME_COL   = "SubmitTime"
BUCKET_COL = "BucketLabel"

# ====== 时间窗口配置（直接在这里改；None 表示不用窗口）======
# 半年=182；三个月=90；全量=None
TRAIN_WINDOW_DAYS: 90 | None = None

# ====== Optuna / 随机种子 ======
N_TRIALS    = 50
RANDOM_SEED = 42

# ====== 时间窗口过滤函数（仅用于训练集）======
def _apply_time_window_training(df: pd.DataFrame,
                                time_col: str = TIME_COL,
                                window_days: int | None = TRAIN_WINDOW_DAYS) -> pd.DataFrame:
    
#    以当前 DataFrame 的最大 SubmitTime 为基准，只保留最近 window_days 天的数据。
#    - window_days=None 或 <=0 时，不做过滤
#    - 仅用于训练集；验证集保持不变
    
    try:
        if not window_days or window_days <= 0:
            return df
        if time_col not in df.columns or len(df) == 0:
            return df
        t = pd.to_datetime(df[time_col], errors="coerce")
        max_t = pd.to_datetime(t.max())
        if pd.isna(max_t):
            return df
        cutoff = max_t - pd.Timedelta(days=window_days)
        out = df.loc[t >= cutoff].copy()
        print(f"[TimeWindow] TRAIN_WINDOW_DAYS={window_days}, cutoff={cutoff.date()}, kept={len(out)}/{len(df)} rows")
        return out
    except Exception as e:
        print(f"[TimeWindow][WARN] failed to apply window: {e}")
        return df

# ====== 工具函数 ======
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
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
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
    df_train = _apply_time_window_training(df_train)  # <<< 新增：训练集时间窗口过滤

    df_val   = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame(columns=df_train.columns)

    df_train = df_train.dropna(subset=[TARGET_COL])
    df_val   = df_val.dropna(subset=[TARGET_COL])

    drop_cols = (TARGET_COL, TIME_COL, BUCKET_COL) if BUCKET_COL in df_train.columns else (TARGET_COL, TIME_COL)
    feature_cols = _numeric_features(df_train, drop_cols=drop_cols)
    if not feature_cols:
        print(f"[WARN] PID {pid}: No numeric features after filtering. Skipping.")
        continue
    _maybe_save_features(feature_cols)

    X_tr_full, y_tr_full = df_train[feature_cols], df_train[TARGET_COL]

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
        if X_eval is not None:
            model.fit(X_tr_full, y_tr_full,
                      eval_set=[(X_eval, y_eval)], eval_metric="rmse",
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            pred = model.predict(X_eval, num_iteration=model.best_iteration_)
            return float(np.sqrt(mean_squared_error(y_eval, pred)))
        else:
            model.set_params(n_estimators=1000)
            model.fit(X_tr_full, y_tr_full)
            pred_tr = model.predict(X_tr_full)
            return float(np.sqrt(mean_squared_error(y_tr_full, pred_tr)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    best_params = study.best_params
    best_params.update({"n_estimators":3000 if X_eval is not None else 1000, "random_state":RANDOM_SEED})

    final_model = LGBMRegressor(**best_params)
    if X_eval is not None:
        final_model.fit(X_tr_full, y_tr_full,
                        eval_set=[(X_eval, y_eval)], eval_metric="rmse",
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        eval_pred = final_model.predict(X_eval, num_iteration=final_model.best_iteration_)
        ov = overall_metrics(y_eval, eval_pred)
    else:
        final_model.fit(X_tr_full, y_tr_full)
        ov = {"RMSE":np.nan,"MAE":np.nan,"Bias":np.nan,"R2":np.nan}

    model_path = os.path.join(MODEL_DIR, f"lgb_model_pid{pid}_optuna.joblib")
    dump(final_model, model_path)

    summary_rows.append({
        "ProgramID_encoded": pid,
        "TrainSize": len(df_train),
        "ValSize": len(df_val),
        "EvalMode": eval_mode,
        **ov,
        "ModelPath": model_path,
        "BestParams": json.dumps(best_params),
    })
    print(f"PID {pid} [{eval_mode}] | RMSE:{ov['RMSE']:.4f} | MAE:{ov['MAE']:.4f} | R2:{ov['R2']:.4f}")

# ====== Others ======
print("\n>> Training fallback model for Others (LGB)…")
df_all_tr = pd.read_csv(ALL_TRAIN)
df_all_tr = _apply_time_window_training(df_all_tr)  # <<< 新增：训练集时间窗口过滤

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
                "num_leaves": trial.suggest_int("num_leaves", 16, 512),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
                "n_estimators": 3000, "random_state": RANDOM_SEED,
            }
            m = LGBMRegressor(**params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            pred = m.predict(Xva, num_iteration=m.best_iteration_)
            return float(np.sqrt(mean_squared_error(yva, pred)))

        study_o = optuna.create_study(direction="minimize"); study_o.optimize(obj_others, n_trials=N_TRIALS)
        best_o = study_o.best_params
        best_o.update({"n_estimators":3000,"random_state":RANDOM_SEED})
        model_o = LGBMRegressor(**best_o)
        model_o.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="rmse",
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        pred_o = model_o.predict(Xva, num_iteration=model_o.best_iteration_)
        rmse_o = float(np.sqrt(mean_squared_error(yva, pred_o)))
        mae_o  = float(mean_absolute_error(yva, pred_o))
    else:
        model_o = LGBMRegressor(n_estimators=1000, random_state=RANDOM_SEED)
        model_o.fit(Xtr, ytr)
        rmse_o = np.nan; mae_o = np.nan

    model_o_path = os.path.join(MODEL_DIR, "lgb_model_others_optuna.joblib")
    dump(model_o, model_o_path)
    summary_rows.append({
        "ProgramID_encoded":"Others","TrainSize":len(df_others_tr),"ValSize":len(df_others_va),
        "EvalMode":"val" if len(df_others_va)>0 else "train_only",
        "RMSE":rmse_o,"MAE":mae_o,"Bias":np.nan,"R2":np.nan,
        "ModelPath":model_o_path,"BestParams":json.dumps(best_o) if len(df_others_va)>0 else "{}"
    })
else:
    print("[SKIP] Others: no training data.")

pd.DataFrame(summary_rows).to_csv(os.path.join(REPORT_DIR, "v3_lgb_90_summary.csv"), index=False)
print("Saved summary to:", os.path.join(REPORT_DIR, "v3_lgb_90_summary.csv"))
