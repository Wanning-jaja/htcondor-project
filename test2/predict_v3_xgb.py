# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from joblib import load

warnings.filterwarnings("ignore", category=UserWarning)

# ===== PATHS =====
INPUT_CSV   = "/home/master/wzheng/projects/model_training/data/40val.csv"
MODEL_DIR   = "/home/master/wzheng/projects/test2/models/v3_182_xgb"   # ← 改成本次实验的模型目录
TOPN_JSON   = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"
OUTPUT_PRED = "/home/master/wzheng/projects/test2/preds/predictions_v3_182_xgb.csv"
# 与模型目录绑定，避免不同实验互相影响
FEATURES_JSON = os.path.join(MODEL_DIR, "features.json")

os.makedirs(os.path.dirname(OUTPUT_PRED), exist_ok=True)

# ===== COLUMNS =====
PID_COL    = "ProgramID_encoded"
TIME_COL   = "SubmitTime"
TARGET_COL = "RemoteWallClockTime"

# ===== Helpers =====
def _load_features(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "feature_cols" in obj:
        return list(obj["feature_cols"])
    if isinstance(obj, list):
        return list(obj)
    return []

def _numeric_features(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> list:
    drop = set(drop_cols)
    ban = {
        "GlobalJobId", "Owner", "OwnerGroup", "Queue",
        "ProgramID", "ProgramName", "ProgramPath4", "ProgramPath",
        "Cmd", "Arguments", "Arguments_merged",
    }
    drop |= ban
    cols = [c for c in df.columns if c not in drop]
    return df[cols].select_dtypes(include=[np.number]).columns.tolist()

def _ensure_feat_matrix(df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    X = []
    for c in feat_cols:
        if c in df.columns:
            X.append(pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy())
        else:
            X.append(np.zeros(len(df), dtype=float))
    return np.vstack(X).T if X else np.zeros((len(df), 0))

def _load_model(path: str):
    return load(path) if (isinstance(path, str) and os.path.exists(path)) else None

def _xgb_predict(model, Xi: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict(Xi), dtype=float)  # sklearn XGBRegressor
        except Exception:
            pass
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(Xi)
        return np.asarray(model.predict(dm), dtype=float)      # Booster
    except Exception as e:
        raise RuntimeError(f"XGB predict failed: {e}")

def _get_model_feature_names_xgb(model) -> List[str] | None:
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) > 0:
        return list(names)
    get_booster = getattr(model, "get_booster", None)
    if callable(get_booster):
        try:
            booster = model.get_booster()
            if hasattr(booster, "feature_names") and booster.feature_names:
                return list(booster.feature_names)
        except Exception:
            return None
    if hasattr(model, "feature_names") and model.feature_names:
        return list(model.feature_names)
    return None

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"INPUT_CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if PID_COL not in df.columns:
        raise ValueError(f"Missing column: {PID_COL}")

    if not os.path.exists(TOPN_JSON):
        raise FileNotFoundError(f"TOPN_JSON not found: {TOPN_JSON}")
    with open(TOPN_JSON, "r", encoding="utf-8") as f:
        top_ids = set(int(x) for x in json.load(f))

    # 先尝试用模型里的“训练期特征名”
    others_model_path = os.path.join(MODEL_DIR, "xgb_model_others_optuna.joblib")
    model_for_names = _load_model(others_model_path)
    if model_for_names is None:
        for pid in list(top_ids)[:200]:
            p = os.path.join(MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib")
            model_for_names = _load_model(p)
            if model_for_names is not None:
                break
    model_feature_names = _get_model_feature_names_xgb(model_for_names) if model_for_names is not None else None

    if model_feature_names:
        feat_cols = [c for c in model_feature_names]
        miss = [c for c in feat_cols if c not in df.columns]
        if miss:
            print(f"[WARN] some model feature cols not in INPUT_CSV: {miss} -> will fill 0")
    else:
        # 再尝试用 features.json；仍不行就兜底数值列
        feat_cols = _load_features(FEATURES_JSON)
        if not feat_cols or not set(feat_cols).issubset(df.columns):
            feat_cols = _numeric_features(df, drop_cols=(TARGET_COL, TIME_COL))
            print(f"[WARN] Using fallback numeric features ({len(feat_cols)}).")

    X_all = _ensure_feat_matrix(df, feat_cols)
    pids  = df[PID_COL].astype(int).to_numpy()
    preds = np.full(len(df), np.nan, dtype=float)

    uniq = np.unique(pids)
    n_missing_model = 0
    for pid in uniq:
        idx = np.where(pids == pid)[0]
        Xi  = X_all[idx, :]

        model_path = (
            os.path.join(MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib")
            if pid in top_ids else
            os.path.join(MODEL_DIR, "xgb_model_others_optuna.joblib")
        )
        m = _load_model(model_path)
        if m is None:
            print(f"[MISS MODEL] {model_path}")
            n_missing_model += len(idx)
            continue

        yi = _xgb_predict(m, Xi)
        preds[idx] = yi.astype(float)

    base_cols = [PID_COL] + ([TIME_COL] if TIME_COL in df.columns else [])
    out = df[base_cols].copy()
    if TARGET_COL in df.columns:
        out[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    out["PredictedRemoteWallClockTime"] = pd.to_numeric(preds, errors="coerce")
    out = out[np.isfinite(out["PredictedRemoteWallClockTime"])]
    out.to_csv(OUTPUT_PRED, index=False)

    print(f"[OK] Saved XGB regression predictions -> {OUTPUT_PRED}")
    print(f"[STATS] rows={len(df)}, predicted={np.isfinite(preds).sum()}, missing_model_rows={n_missing_model}")
    print(f"[INFO] n_features={len(feat_cols)}")

if __name__ == "__main__":
    main()
