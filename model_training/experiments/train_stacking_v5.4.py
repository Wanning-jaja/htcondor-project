# -*- coding: utf-8 -*-

#train_stacking_v5.4.py  (fixed model dirs, no summary dependency)

#- 直接按固定模板加载模型：
#  XGB: /home/master/wzheng/projects/model_training/models/v5.4_xgb/xgb_model_pid{pid}_optuna.joblib
#  LGB: /home/master/wzheng/projects/model_training/models/v5.4_lgb/lgb_model_pid{pid}_optuna.joblib
#- 每个 PID 的评估集：优先使用 val；若无，则用训练集尾部做 5%（>=200、<=20000）内部留出
#- 候选策略：xgb_only / lgb_only / weighted(w∈{0,0.1,...,1})
#- 目标函数：score = RMSE + LAMBDA_BUCKET * mean(|pred_bucket - true_bucket|)
#- 输出：
#  /models/v5.4_ensemble/pid_ensemble_v5.4.csv
#  /reports/v5.4_ensemble/v5.4_ensemble_evaluation_summary.csv

from __future__ import annotations
import os, json, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ====== 固定路径 ======
SPLIT_DIR     = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON     = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"

XGB_MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v5.4_xgb"
LGB_MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v5.4_lgb"

ENSEMBLE_DIR  = "/home/master/wzheng/projects/model_training/models/v5.4_ensemble"
REPORT_DIR    = "/home/master/wzheng/projects/model_training/reports/v5.4_ensemble"
FEATURES_JSON = "/home/master/wzheng/projects/model_training/models/v5.4_features.json"

os.makedirs(ENSEMBLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ====== 字段 & 桶定义（5 桶：<1h, 1–3h, 3–6h, 6–12h, ≥12h）======
TARGET_COL = "RemoteWallClockTime"
TIME_COL   = "SubmitTime"
BUCKET_COL = "BucketLabel"

#_BUCKET_EDGES  = [3600, 10800, 21600, 43200]
BUCKET_EDGES_MINUTES = [0, 30, 60, 180, 360, 720, float("inf")]
def _minutes_to_seconds_edges(mins):
    return [float("inf") if m == float("inf") else int(m*60) for m in mins]
_BUCKET_EDGES = _minutes_to_seconds_edges(BUCKET_EDGES_MINUTES)

LAMBDA_BUCKET  = 0.20  # 桶惩罚权重

# ====== 工具函数 ======
def _bucketize(a): 
    return np.digitize(np.asarray(a, float), _BUCKET_EDGES, right=False)

def _bucket_penalty(y, p):
    tb, pb = _bucketize(y), _bucketize(p)
    return np.mean(np.abs(pb - tb))

def _score(y_true, y_pred) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bpen = float(_bucket_penalty(y_true, y_pred))
    return rmse + LAMBDA_BUCKET * bpen, rmse, bpen

def _metrics(y, p) -> Dict[str, float]:
    y = np.asarray(y, float); p = np.asarray(p, float)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae  = float(np.mean(np.abs(p - y)))
    mape = float(np.mean(np.abs((p - y)/np.clip(np.abs(y),1e-9,None)))*100.0) if np.any(np.abs(y)>1e-9) else np.nan
    bias = float(np.mean(p - y))
    try:
        r2 = float(r2_score(y, p))
    except Exception:
        r2 = np.nan
    tb, pb = _bucketize(y), _bucketize(p)
    acc  = float(np.mean(tb == pb))
    acc1 = float(np.mean(np.abs(tb - pb) <= 1))
    under= float(np.mean(pb < tb))
    over = float(np.mean(pb > tb))
    hit  = float(np.mean(pb == tb))
    return {"RMSE":rmse,"MAE":mae,"MAPE(%)":mape,"Bias":bias,"R2":r2,
            "Accuracy":acc,"Accuracy+-1":acc1,
            "under_rate":under,"over_rate":over,"hit_rate":hit}

def _load_features(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FEATURES_JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "feature_cols" in obj:
        feats = list(obj["feature_cols"])
    else:
        feats = list(obj)
    if not feats:
        raise ValueError("feature_cols is empty in FEATURES_JSON.")
    return feats

def _feat_mat(df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    X = []
    for c in feat_cols:
        if c in df.columns:
            X.append(df[c].astype(float).to_numpy())
        else:
            if not hasattr(_feat_mat, "_warned"):
                print(f"[WARN] feature '{c}' missing in DF; fill 0. (silent afterwards)")
                _feat_mat._warned = True
            X.append(np.zeros(len(df), dtype=float))
    return np.vstack(X).T if X else np.zeros((len(df), 0))

def _internal_holdout(df_sorted: pd.DataFrame, ratio=0.05, min_ho=200, max_ho=20000):
#从按时间排好序的 df 中切尾部一段作为评估集
    n = len(df_sorted)
    n_ho = max(min(int(n*ratio), max_ho), min_ho)
    if n <= n_ho:
        return df_sorted.copy(), df_sorted.copy()
    return df_sorted.iloc[:n-n_ho].copy(), df_sorted.iloc[n-n_ho:].copy()

def _model_path(pid: int, kind: str) -> str:
#按固定模板拼模型路径；kind ∈ {'xgb','lgb'}
    if kind == "xgb":
        return os.path.join(XGB_MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib")
    else:
        return os.path.join(LGB_MODEL_DIR, f"lgb_model_pid{pid}_optuna.joblib")

def _load_model(path: str):
    if isinstance(path, str) and os.path.exists(path):
        try:
            return load(path)
        except Exception as e:
            print(f"[WARN] failed to load model: {path} ({e})")
            return None
    return None

# ====== 主流程 ======
def main():
    print("Start stacking_v5.4 (fixed model dirs)")
    feats = _load_features(FEATURES_JSON)
    print(f"[INFO] feature count = {len(feats)}")

    with open(TOPN_JSON, "r") as f:
        pids = [int(x) for x in json.load(f)]

    ensemble_rows: List[Dict] = []
    overall_collect: List[Dict] = []

    for pid in pids + ["Others"]:
        if pid != "Others":
            tr_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
            va_path = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")
            if not os.path.exists(tr_path):
                print(f"[SKIP] PID {pid}: train file not found.")
                continue

            df_tr = pd.read_csv(tr_path).dropna(subset=[TARGET_COL])
            df_va = pd.read_csv(va_path).dropna(subset=[TARGET_COL]) if os.path.exists(va_path) else pd.DataFrame(columns=df_tr.columns)

            if len(df_va) == 0:
                df_sorted = df_tr.sort_values(TIME_COL).reset_index(drop=True)
                df_fit, df_eval = _internal_holdout(df_sorted, ratio=0.05, min_ho=200, max_ho=20000)
                eval_mode = "holdout"
            else:
                df_fit, df_eval = df_tr.copy(), df_va.copy()
                eval_mode = "val"

            X_eval = _feat_mat(df_eval, feats)
            y_eval = df_eval[TARGET_COL].to_numpy(float)

            x_path = _model_path(int(pid), "xgb")
            l_path = _model_path(int(pid), "lgb")
            if not os.path.exists(x_path):
                print(f"[WARN] XGB model missing for PID {pid}: {x_path}")
            if not os.path.exists(l_path):
                print(f"[WARN] LGB model missing for PID {pid}: {l_path}")

            xgb_model = _load_model(x_path)
            lgb_model = _load_model(l_path)

            if (xgb_model is None) and (lgb_model is None):
                print(f"[WARN] PID {pid}: no models available, skipping. (eval_mode={eval_mode})")
                continue

            # 候选策略
            cand = []
            if xgb_model is not None:
                px = xgb_model.predict(X_eval)
                sc, rmse, bpen = _score(y_eval, px)
                cand.append(("xgb_only", 1.0, 0.0, sc, rmse, bpen, px))
            if lgb_model is not None:
                pl = lgb_model.predict(X_eval)
                sc, rmse, bpen = _score(y_eval, pl)
                cand.append(("lgb_only", 0.0, 1.0, sc, rmse, bpen, pl))
            if (xgb_model is not None) and (lgb_model is not None):
                px = xgb_model.predict(X_eval)
                pl = lgb_model.predict(X_eval)
                for w in np.linspace(0.0, 1.0, 11):
                    p = w*px + (1.0-w)*pl
                    sc, rmse, bpen = _score(y_eval, p)
                    cand.append(("weighted", float(w), float(1.0-w), sc, rmse, bpen, p))

            cand.sort(key=lambda t: t[3])
            best = cand[0]
            strategy, w_xgb, w_lgb, best_score, best_rmse, best_bpen, best_pred = best

            mets = _metrics(y_eval, best_pred)
            overall_collect.append(mets)

            ensemble_rows.append({
                "ProgramID_encoded": pid,
                "strategy": strategy,
                "w_xgb": w_xgb, "w_lgb": w_lgb,
                "xgb_model_path": x_path if os.path.exists(x_path) else "",
                "lgb_model_path": l_path if os.path.exists(l_path) else "",
                "best_score": best_score, "rmse": best_rmse, "bucket_pen": best_bpen,
                "safety_shift_sec": 0.0,
                "EvalMode": eval_mode
            })

        else:

            # === Others：像普通 PID 一样选择最优策略，并把模型路径写进配置 ===
            tr_path = os.path.join(SPLIT_DIR, "train_topOthers.csv")
            va_path = os.path.join(SPLIT_DIR, "val_topOthers.csv")
            if not os.path.exists(tr_path):
                print("[SKIP] Others: train file not found.")
                continue

            df_tr  = pd.read_csv(tr_path).dropna(subset=[TARGET_COL])
            df_va  = pd.read_csv(va_path).dropna(subset=[TARGET_COL]) if os.path.exists(va_path) else pd.DataFrame(columns=df_tr.columns)

            if len(df_va) == 0:
                df_sorted = df_tr.sort_values(TIME_COL).reset_index(drop=True)
                df_fit, df_eval = _internal_holdout(df_sorted, ratio=0.05, min_ho=200, max_ho=20000)
                eval_mode = "holdout"
            else:
                df_fit, df_eval = df_tr.copy(), df_va.copy()
                eval_mode = "val"

            X_eval = _feat_mat(df_eval, feats)
            y_eval = df_eval[TARGET_COL].to_numpy(float)

            # 后备模型固定路径（和你的训练脚本一致）
            x_path = os.path.join(XGB_MODEL_DIR, "xgb_model_others_optuna.joblib")
            l_path = os.path.join(LGB_MODEL_DIR, "lgb_model_others_optuna.joblib")
            if not os.path.exists(x_path):
                print(f"[WARN] Others XGB model missing: {x_path}")
            if not os.path.exists(l_path):
                print(f"[WARN] Others LGB model missing: {l_path}")

            xgb_model = _load_model(x_path)
            lgb_model = _load_model(l_path)

            # 和上面 TopN 的候选流程保持一致
            cand = []
            if xgb_model is not None:
                px = xgb_model.predict(X_eval)
                sc, rmse, bpen = _score(y_eval, px)
                cand.append(("xgb_only", 1.0, 0.0, sc, rmse, bpen, px))
            if lgb_model is not None:
                pl = lgb_model.predict(X_eval)
                sc, rmse, bpen = _score(y_eval, pl)
                cand.append(("lgb_only", 0.0, 1.0, sc, rmse, bpen, pl))
            if (xgb_model is not None) and (lgb_model is not None):
                px = xgb_model.predict(X_eval)
                pl = lgb_model.predict(X_eval)
                for w in np.linspace(0.0, 1.0, 11):
                    p = w*px + (1.0-w)*pl
                    sc, rmse, bpen = _score(y_eval, p)
                    cand.append(("weighted", float(w), float(1.0-w), sc, rmse, bpen, p))

            if not cand:
                print("[WARN] Others: no models available, skip writing row.")
                continue

            cand.sort(key=lambda t: t[3])
            strategy, w_xgb, w_lgb, best_score, best_rmse, best_bpen, best_pred = cand[0]
            mets = _metrics(y_eval, best_pred)
            overall_collect.append(mets)

            ensemble_rows.append({
                "ProgramID_encoded": "Others",
                "strategy": strategy,
                "w_xgb": w_xgb, "w_lgb": w_lgb,
                "xgb_model_path": x_path if os.path.exists(x_path) else "",
                "lgb_model_path": l_path if os.path.exists(l_path) else "",
                "best_score": best_score, "rmse": best_rmse, "bucket_pen": best_bpen,
                "safety_shift_sec": 0.0,
                "EvalMode": eval_mode
            })


    # 保存 per-PID 策略
    cfg_path = os.path.join(ENSEMBLE_DIR, "pid_ensemble_v5.4.csv")
    pd.DataFrame(ensemble_rows).to_csv(cfg_path, index=False)
    print(">>> Ensemble config saved to:", cfg_path)

    # 汇总整体
    if overall_collect:
        dfm = pd.DataFrame(overall_collect)
        ov = {k: float(np.nanmean(dfm[k])) for k in dfm.columns}
    else:
        ov = {k: np.nan for k in ["RMSE","MAE","MAPE(%)","Bias","R2","Accuracy","Accuracy+-1","under_rate","over_rate","hit_rate"]}

    sum_path = os.path.join(REPORT_DIR, "v5.4_ensemble_evaluation_summary.csv")
    pd.DataFrame([ov]).to_csv(sum_path, index=False)
    print(">>> Ensemble evaluation summary saved to:", sum_path)
    print(f"\n=== Ensemble Overall (Eval) Averages ({len(BUCKET_EDGES_MINUTES)-1}-bins) ===")

    for k, v in ov.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
