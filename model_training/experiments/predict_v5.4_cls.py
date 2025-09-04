# -*- coding: utf-8 -*- 
from __future__ import annotations
import os, json, warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from joblib import load

warnings.filterwarnings("ignore", category=UserWarning)

# ===== PATHS =====
SPLIT_DIR     = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON     = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"
XGB_DIR       = "/home/master/wzheng/projects/model_training/models/v5.4_cls_xgb"
#XGB_DIR       = "/home/master/wzheng/projects/model_training/models/v5.5_cls_xgb_upspeed"
LGB_DIR       = "/home/master/wzheng/projects/model_training/models/v5.4_cls_lgb"
FEATURES_JSON = "/home/master/wzheng/projects/model_training/models/v5.4_features.json"

INPUT_CSV     = "/home/master/wzheng/projects/model_training/data/40val.csv"
OUTPUT_CSV    = "/home/master/wzheng/projects/model_training/preds/v5.4_cls_predictions.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ===== 桶定义（与训练/评估一致） =====
BUCKET_EDGES_MINUTES = [0, 30, 60, 180, 360, 720, float("inf")]
def _minutes_to_seconds_edges(mins):
    return [float("inf") if m == float("inf") else int(m*60) for m in mins]
_BUCKET_EDGES  = _minutes_to_seconds_edges(BUCKET_EDGES_MINUTES)
_INTERNAL_BINS = _BUCKET_EDGES[1:-1]
LOWER_BOUNDS   = _BUCKET_EDGES[:-1]
UPPER_BOUNDS   = _BUCKET_EDGES[1:]
N_BUCKETS      = len(LOWER_BOUNDS)

def _bucketize_seconds(arr: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(arr, float), _INTERNAL_BINS, right=False)

def _bucket_mid_seconds(b: int) -> float:
    lo, hi = LOWER_BOUNDS[b], UPPER_BOUNDS[b]
    return (lo + (hi if np.isfinite(hi) else lo*2.0)) / 2.0

# ===== 代价敏感（可选）=====
COST_SENSITIVE_DECISION = False
LAM_UNDER = 0.6
LAM_OVER  = 0.2

def _build_cost_matrix(n_buckets: int, lu: float, lo: float) -> np.ndarray:
    J = np.arange(n_buckets)
    K = np.arange(n_buckets)[:, None]
    under = np.clip(J - K, 0, None)
    over  = np.clip(K - J, 0, None)
    return lu * under + lo * over

# ===== 列名 =====
PID_COL    = "ProgramID_encoded"
TIME_COL   = "SubmitTime"
TARGET_COL = "RemoteWallClockTime"
BUCKET_COL = "BucketLabel"

def _load_features(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return list(obj["feature_cols"]) if isinstance(obj, dict) and "feature_cols" in obj else list(obj)

def _numeric_features(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> list:
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].select_dtypes(include=[np.number]).columns.tolist()

def _ensure_feat_matrix(df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    X = []
    for c in feat_cols:
        if c in df.columns:
            X.append(df[c].astype(float).to_numpy())
        else:
            X.append(np.zeros(len(df), dtype=float))
    return np.vstack(X).T if X else np.zeros((len(df), 0))

def _first_existing(path_list: List[str]) -> str | None:
    for p in path_list:
        if p and os.path.exists(p):
            return p
    return None

def _candidate_paths(base_dir: str, framework: str, pid: int | None, is_top: bool) -> List[str]:
    
#    兼容多种命名：
#    - 新：xgb_cls_pid{pid}_optuna/default/constant.joblib ；others 同理
#    - 旧：xgb_model_pid{pid}_optuna/default/constant.joblib ；others 同理
#    - LGB 同理：lgb_cls_* / lgb_model_*
    
    if framework not in ("xgb", "lgb"):
        return []
    cls_prefix = f"{framework}_cls_pid{pid}" if pid is not None else f"{framework}_cls_others"
    mdl_prefix = f"{framework}_model_pid{pid}" if pid is not None else f"{framework}_model_others"
    return [
        os.path.join(base_dir, f"{cls_prefix}_optuna.joblib"),
        os.path.join(base_dir, f"{cls_prefix}_default.joblib"),
        os.path.join(base_dir, f"{cls_prefix}_constant.joblib"),  # 新增：支持 constant
        os.path.join(base_dir, f"{mdl_prefix}_optuna.joblib"),
        os.path.join(base_dir, f"{mdl_prefix}_default.joblib"),
        os.path.join(base_dir, f"{mdl_prefix}_constant.joblib"),  # 旧前缀的 constant
    ]

def main():
    df = pd.read_csv(INPUT_CSV)
    if PID_COL not in df.columns:
        raise ValueError(f"Missing column: {PID_COL}")

    with open(TOPN_JSON, "r") as f:
        top_ids = set(int(x) for x in json.load(f))

    # 特征列：优先训练端的 features.json
    try:
        feat_cols = _load_features(FEATURES_JSON)
        if not set(feat_cols).issubset(set(df.columns)):
            raise RuntimeError("feature cols missing in input df")
    except Exception:
        feat_cols = _numeric_features(df, drop_cols=(TARGET_COL, TIME_COL, BUCKET_COL))
        print(f"[WARN] Using fallback numeric features ({len(feat_cols)}).")

    X_all = _ensure_feat_matrix(df, feat_cols)
    pids  = df[PID_COL].astype(int).to_numpy()

    # 结果容器（先填 NaN，只有成功预测的才覆盖）
    out_dict: Dict[str, np.ndarray] = {}
    out_dict["pred_bucket_xgb"] = np.full(len(df), np.nan)
    out_dict["pred_bucket_lgb"] = np.full(len(df), np.nan)
    out_dict["PredictedRemoteWallClockTime_xgb"] = np.full(len(df), np.nan)
    out_dict["PredictedRemoteWallClockTime_lgb"] = np.full(len(df), np.nan)

    if COST_SENSITIVE_DECISION:
        COST_MAT = _build_cost_matrix(N_BUCKETS, LAM_UNDER, LAM_OVER)
        out_dict["pred_bucket_xgb_cs"] = np.full(len(df), np.nan)
        out_dict["PredictedRemoteWallClockTime_xgb_cs"] = np.full(len(df), np.nan)
        out_dict["pred_bucket_lgb_cs"] = np.full(len(df), np.nan)
        out_dict["PredictedRemoteWallClockTime_lgb_cs"] = np.full(len(df), np.nan)
    else:
        COST_MAT = None

    # 统计用
    missing_xgb, missing_lgb = [], []

    # 分 PID 预测
    for pid in np.unique(pids):
        idx = np.where(pids == pid)[0]
        Xi  = X_all[idx, :]

        is_top = (pid in top_ids)
        # 找 XGB 模型
        x_path = _first_existing(_candidate_paths(XGB_DIR, "xgb", pid if is_top else None, is_top))
        if x_path is None:
            missing_xgb.append(pid)
        else:
            mx = load(x_path)
            proba_x = None
            if hasattr(mx, "predict_proba"):
                try:
                    proba_x = mx.predict_proba(Xi)
                    b = np.argmax(proba_x, axis=1)
                except Exception:
                    proba_x = None
                    b = mx.predict(Xi).astype(int)
            else:
                b = mx.predict(Xi).astype(int)

            out_dict["pred_bucket_xgb"][idx] = b
            out_dict["PredictedRemoteWallClockTime_xgb"][idx] = np.array([_bucket_mid_seconds(int(bb)) for bb in b], dtype=float)

            if COST_MAT is not None and proba_x is not None and proba_x.shape[1] == N_BUCKETS:
                exp_cost = proba_x @ COST_MAT.T
                b_cs = np.argmin(exp_cost, axis=1)
                out_dict["pred_bucket_xgb_cs"][idx] = b_cs
                out_dict["PredictedRemoteWallClockTime_xgb_cs"][idx] = np.array([_bucket_mid_seconds(int(bb)) for bb in b_cs], dtype=float)

        # 找 LGB 模型
        l_path = _first_existing(_candidate_paths(LGB_DIR, "lgb", pid if is_top else None, is_top))
        if l_path is None:
            missing_lgb.append(pid)
        else:
            ml = load(l_path)
            proba_l = None
            if hasattr(ml, "predict_proba"):
                try:
                    proba_l = ml.predict_proba(Xi)
                    b = np.argmax(proba_l, axis=1)
                except Exception:
                    proba_l = None
                    b = ml.predict(Xi).astype(int)
            else:
                b = ml.predict(Xi).astype(int)

            out_dict["pred_bucket_lgb"][idx] = b
            out_dict["PredictedRemoteWallClockTime_lgb"][idx] = np.array([_bucket_mid_seconds(int(bb)) for bb in b], dtype=float)

            if COST_MAT is not None and proba_l is not None and proba_l.shape[1] == N_BUCKETS:
                exp_cost = proba_l @ COST_MAT.T
                b_cs = np.argmin(exp_cost, axis=1)
                out_dict["pred_bucket_lgb_cs"][idx] = b_cs
                out_dict["PredictedRemoteWallClockTime_lgb_cs"][idx] = np.array([_bucket_mid_seconds(int(bb)) for bb in b_cs], dtype=float)

    # 组织输出基础列
    base_cols = [PID_COL, TIME_COL] if TIME_COL in df.columns else [PID_COL]
    if BUCKET_COL in df.columns:
        base_cols.append(BUCKET_COL)
    out = df[base_cols].copy()

    # 写入预测列
    for k, v in out_dict.items():
        out[k] = v

    # 如果原表带真值，也一并带出
    if TARGET_COL in df.columns:
        out[TARGET_COL] = df[TARGET_COL]

    # true_bucket 兜底
    if "true_bucket" not in out.columns:
        if BUCKET_COL in out.columns:
            out["true_bucket"] = out[BUCKET_COL].astype("Int64")
        elif TARGET_COL in out.columns:
            out["true_bucket"] = _bucketize_seconds(out[TARGET_COL].to_numpy(float)).astype("Int64")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved classification predictions -> {OUTPUT_CSV}")

    # 友好提示
    if missing_xgb:
        print(f"[WARN] XGB model not found for {len(missing_xgb)} PID(s), e.g. {missing_xgb[:10]}")
    if missing_lgb:
        print(f"[WARN] LGB model not found for {len(missing_lgb)} PID(s), e.g. {missing_lgb[:10]}")

if __name__ == "__main__":
    main()
