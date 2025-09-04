# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from joblib import load

warnings.filterwarnings("ignore", category=UserWarning)

# ===== PATHS（按你的现有目录）=====
TOPN_JSON       = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"
MODEL_DIR       = "/home/master/wzheng/projects/model_training/models/v3.3_lgb"
INPUT_CSV       = "/home/master/wzheng/projects/model_training/data/40val.csv"
OUTPUT_CSV      = "/home/master/wzheng/projects/model_training/preds/predictions_v3.3_lgb.csv"
# 如果训练端保存了特征清单，指向它；没有也没关系（会自动回退为数值列）
FEATURES_JSON   = "/home/master/wzheng/projects/model_training/models/v3.3_features.json"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ===== COLUMNS（与分类脚本一致的命名约定）=====
PID_COL    = "ProgramID_encoded"
TIME_COL   = "SubmitTime"
TARGET_COL = "RemoteWallClockTime"

# ===== 工具函数 =====
def _load_features(path: str) -> List[str]:
    # 优先读取训练端落盘的特征列清单（list 或 {'feature_cols': [...] }）
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
    # 兜底：用数值列，去掉 y / 时间 / 标签 / 明显无关的标识列
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
    # 将特征列按顺序拼成 ndarray；缺失列用 0 兜底
    X = []
    for c in feat_cols:
        if c in df.columns:
            X.append(pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy())
        else:
            X.append(np.zeros(len(df), dtype=float))
    return np.vstack(X).T if X else np.zeros((len(df), 0))

def _load_model(path: str):
    return load(path) if (isinstance(path, str) and os.path.exists(path)) else None

def _get_model_feature_names(model) -> List[str] | None:
    # 从 LightGBM/sklearn 模型对象获取训练时的特征名（若可用）
    names = getattr(model, "feature_name_", None)
    if names:
        return list(names)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            return list(booster.feature_name())
        except Exception:
            return None
    return None

def main():
    # ===== 读取数据 =====
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"INPUT_CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if PID_COL not in df.columns:
        raise ValueError(f"Missing column: {PID_COL}")

    # ===== 读取 TopN 清单 =====
    if not os.path.exists(TOPN_JSON):
        raise FileNotFoundError(f"TOPN_JSON not found: {TOPN_JSON}")
    with open(TOPN_JSON, "r", encoding="utf-8") as f:
        top_ids = set(int(x) for x in json.load(f))

    # ===== 特征列：优先用训练端 FEATURES_JSON；不全则回退为数值列 =====
    feat_cols = _load_features(FEATURES_JSON)
    if not feat_cols or not set(feat_cols).issubset(df.columns):
        feat_cols = _numeric_features(df, drop_cols=(TARGET_COL, TIME_COL))
        print(f"[WARN] Using fallback numeric features ({len(feat_cols)}).")

    # ===== 先加载一个模型，尽量拿到“训练期特征名”（以保证顺序/集合一致）=====
    others_model_path = os.path.join(MODEL_DIR, "lgb_model_others_optuna.joblib")
    model_for_names = _load_model(others_model_path)
    if model_for_names is None:
        # 找一个 TopN 的模型兜底
        for pid in list(top_ids)[:200]:
            p = os.path.join(MODEL_DIR, f"lgb_model_pid{pid}_optuna.joblib")
            model_for_names = _load_model(p)
            if model_for_names is not None:
                break
    model_feature_names = _get_model_feature_names(model_for_names) if model_for_names is not None else None
    if model_feature_names:
        # 用模型记录的列名覆盖 feat_cols（更安全，确保对齐）
        feat_cols = [c for c in model_feature_names]
        miss = [c for c in feat_cols if c not in df.columns]
        if miss:
            print(f"[WARN] some model feature cols not in INPUT_CSV: {miss} -> will fill 0")

    # ===== 准备特征矩阵和 pid 数组 =====
    X_all = _ensure_feat_matrix(df, feat_cols)
    pids  = df[PID_COL].astype(int).to_numpy()

    # ===== 容器 =====
    pred = np.full(len(df), np.nan, dtype=float)

    # ===== 分 PID 预测（TopN 用专模；其余用 others）=====
    uniq = np.unique(pids)
    n_missing_model = 0
    for pid in uniq:
        idx = np.where(pids == pid)[0]
        Xi  = X_all[idx, :]

        model_path = (
            os.path.join(MODEL_DIR, f"lgb_model_pid{pid}_optuna.joblib")
            if pid in top_ids else
            os.path.join(MODEL_DIR, "lgb_model_others_optuna.joblib")
        )
        m = _load_model(model_path)
        if m is None:
            print(f"[MISS MODEL] {model_path}")
            n_missing_model += len(idx)
            continue

        # LightGBM/sklearn 统一 predict
        yi = m.predict(Xi)
        pred[idx] = yi.astype(float)

    # ===== 组织输出并保存 =====
    base_cols = [PID_COL] + ([TIME_COL] if TIME_COL in df.columns else [])
    out = df[base_cols].copy()

    # 评估脚本需要的真值列名（若输入里本就有）
    if TARGET_COL in df.columns:
        out[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # 评估脚本需要的预测列名
    out["PredictedRemoteWallClockTime"] = pd.to_numeric(pred, errors="coerce")

    # 丢掉没预测出来的行（NaN/inf）
    out = out[np.isfinite(out["PredictedRemoteWallClockTime"])]

    out.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Saved regression predictions -> {OUTPUT_CSV}")
    print(f"[STATS] rows={len(df)}, predicted={np.isfinite(pred).sum()}, missing_model_rows={n_missing_model}")
    print(f"[INFO] n_features={len(feat_cols)}")

if __name__ == "__main__":
    main()
