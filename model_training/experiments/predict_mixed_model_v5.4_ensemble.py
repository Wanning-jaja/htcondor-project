# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np, pandas as pd
from joblib import load
warnings.filterwarnings("ignore", category=UserWarning)

# ===== PATHS =====
ENSEMBLE_CFG = "/home/master/wzheng/projects/model_training/models/v5.4_ensemble/pid_ensemble_v5.4.csv"
INPUT_CSV    = "/home/master/wzheng/projects/model_training/data/40val.csv"  # TODO
OUTPUT_CSV   = "/home/master/wzheng/projects/model_training/preds/v5.4_ensemble_predictions.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

FEATURES_JSON = "/home/master/wzheng/projects/model_training/models/v5.4_features.json"
TOPN_JSON     = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"

TARGET_COL = "RemoteWallClockTime"
TIME_COL   = "SubmitTime"
BUCKET_COL = "BucketLabel"
PID_COL    = "ProgramID_encoded"

# ===== 后处理开关 （方案B经测试为最佳） =====

APPLY_SAFETY_SHIFT = False      # 是否加安全裕量
CLIP_TO_BUCKET     = True      # 是否把预测裁回桶内
CLIP_STRATEGY      = "clip"     # "clip" 或 "mid"


#_BUCKET_EDGES = [3600, 10800, 21600, 43200]
# 以“分钟”为单位定义桶边界，内部自动换秒（保持 [left, right) 口径）
BUCKET_EDGES_MINUTES = [0, 30, 60, 180, 360, 720, float("inf")]  # 例：<30m, 30–60m, 1–3h, 3–6h, 6–12h, ≥12h
def _minutes_to_seconds_edges(mins):
    return [float("inf") if m == float("inf") else int(m*60) for m in mins]

_BUCKET_EDGES = _minutes_to_seconds_edges(BUCKET_EDGES_MINUTES)
_INTERNAL_BINS = _BUCKET_EDGES[1:-1]
def numeric_features(df: pd.DataFrame, drop_cols: tuple) -> List[str]:
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].select_dtypes(include=[np.number]).columns.tolist()

def load_feature_list(path: str) -> Optional[List[str]]:
    if os.path.exists(path):
        try:
            with open(path,"r") as f: data = json.load(f)
            return list(data["feature_cols"]) if isinstance(data,dict) and "feature_cols" in data else list(data)
        except Exception as e:
            print(f"[WARN] FEATURES_JSON parse error: {e}")
    return None

def ensure_feature_matrix(df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    X_list = []
    for c in feat_cols:
        if c in df.columns:
            X_list.append(df[c].astype(float).to_numpy())
        else:
            print(f"[WARN] missing feature '{c}', fill 0.")
            X_list.append(np.zeros(len(df), dtype=float))
    return np.vstack(X_list).T

def _bucketize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.digitize(a, _INTERNAL_BINS, right=False)  # -> 0..K-1

def _bucket_bounds(b: int) -> Tuple[float,float]:
    # b in [0, K-1]
    return _BUCKET_EDGES[b], _BUCKET_EDGES[b+1]
def post_clip(preds: np.ndarray, how="mid"):
    preds = np.asarray(preds, dtype=float)
    b = _bucketize(preds); out = preds.copy()
    for i,bi in enumerate(b):
        lo,hi = _bucket_bounds(int(bi))
        if how=="mid":
            out[i] = (lo + (hi if np.isfinite(hi) else lo*2.0))/2.0
        else:
            out[i] = np.clip(out[i], lo, (hi-1e-6) if np.isfinite(hi) else np.inf)
    return out

# load ensemble cfg
cfg = pd.read_csv(ENSEMBLE_CFG)
cfg_map: Dict[int, pd.Series] = {int(r.ProgramID_encoded): r for _,r in cfg.iterrows() if r.ProgramID_encoded!="Others"}
others_cfg = cfg[cfg.ProgramID_encoded=="Others"].head(1)
others_xgb_path = others_cfg["xgb_model_path"].iloc[0] if not others_cfg.empty else None
others_lgb_path = others_cfg["lgb_model_path"].iloc[0] if not others_cfg.empty else None
others_shift    = float(others_cfg["safety_shift_sec"].iloc[0]) if ("safety_shift_sec" in others_cfg.columns and not others_cfg.empty) else 0.0

model_cache: Dict[str, object] = {}
def get_model(path: Optional[str]):
    if not path or not isinstance(path,str): return None
    if not os.path.exists(path):
        print(f"[WARN] model missing: {path}")
        return None
    if path not in model_cache:
        model_cache[path] = load(path)
    return model_cache[path]

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"[INFO] input rows={len(df)}; columns={len(df.columns)}")
    if PID_COL not in df.columns: raise ValueError(f"Missing '{PID_COL}'")

    # features
    feats = load_feature_list(FEATURES_JSON)
    if feats is None:
        drop_cols = (TARGET_COL, TIME_COL, BUCKET_COL) if BUCKET_COL in df.columns else (TARGET_COL, TIME_COL)
        feats = numeric_features(df, drop_cols)
        print(f"[WARN] no FEATURES_JSON, fallback numeric cols ({len(feats)})")
    else:
        print(f"[INFO] locked features from FEATURES_JSON ({len(feats)})")
    X_all = ensure_feature_matrix(df, feats)

    pids = df[PID_COL].astype(int).to_numpy()
    uniq = np.unique(pids)
    yhat = np.full(len(df), np.nan, dtype=float)

    for pid in uniq:
        idx = np.where(pids==pid)[0]
        Xp  = X_all[idx,:]

        if pid in cfg_map:
            r = cfg_map[pid]
            strategy = str(r.strategy)
            w_xgb, w_lgb = float(r.w_xgb), float(r.w_lgb)
            xgb_path = r.xgb_model_path if isinstance(r.xgb_model_path,str) and len(r.xgb_model_path) else None
            lgb_path = r.lgb_model_path if isinstance(r.lgb_model_path,str) and len(r.lgb_model_path) else None
            safety_shift = float(r["safety_shift_sec"]) if "safety_shift_sec" in r.index else 0.0
        else:
            strategy = "weighted"; w_xgb, w_lgb = 0.5, 0.5
            xgb_path, lgb_path = others_xgb_path, others_lgb_path
            safety_shift = others_shift

        # batch predict
        if strategy == "xgb_only":
            mx = get_model(xgb_path)
            pred = mx.predict(Xp).astype(float) if mx is not None else np.full(len(idx), np.nan)
        elif strategy == "lgb_only":
            ml = get_model(lgb_path)
            pred = ml.predict(Xp).astype(float) if ml is not None else np.full(len(idx), np.nan)
        else:
            mx, ml = get_model(xgb_path), get_model(lgb_path)
            px = mx.predict(Xp).astype(float) if mx is not None else None
            pl = ml.predict(Xp).astype(float) if ml is not None else None
            if (px is not None) and (pl is not None):
                pred = w_xgb*px + w_lgb*pl
            elif px is not None:
                pred = px
            elif pl is not None:
                pred = pl
            else:
                pred = np.full(len(idx), np.nan)

        # 方向敏感：应用安全裕量（偏向“宁可略高估也别低估”）
        if APPLY_SAFETY_SHIFT and safety_shift>0:
            pred = pred + safety_shift

        # 桶裁剪/映射
        if CLIP_TO_BUCKET:
            pred = post_clip(pred, how=CLIP_STRATEGY)

        yhat[idx] = pred

    out = df.copy()
    out["PredictedRemoteWallClockTime"] = yhat

    # 如果在验证集上（包含真实目标），输出你要的方向字段
    if TARGET_COL in df.columns:
        y_true = df[TARGET_COL].to_numpy(dtype=float)
        tb, pb = _bucketize(y_true), _bucketize(yhat)
        out["true_bucket"] = tb
        out["pred_bucket"] = pb
        out["hit"]   = (tb==pb).astype(int)
        diff = pb - tb
        out["deviation"] = diff     # +1=高估1桶，-2=低估2桶...
        out["direction"] = np.where(diff==0, "hit", np.where(diff>0, "over", "under"))

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved predictions to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
