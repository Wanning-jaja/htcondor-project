# -*- coding: utf-8 -*-
# evaluate_5.4.py �� ABCD ����ʵ�飺����� Hit/Under/Over/Total/Hit Rate(%)
from __future__ import annotations
import os, json, warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ========= ·�� =========
INPUT_CSV  = "/home/master/wzheng/projects/model_training/preds/v5.4_cls_predictions.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/evaluation/ABCD/v5.4A_cls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= �ֶ� =========
Y_TRUE_COL  = "RemoteWallClockTime"
PID_COL     = "ProgramID_encoded"
TRUE_BKT    = "true_bucket"

# Ԥ���У�����Ҫ�� XGB/LGB �ġ��롱�����ӦͰ�У�
SEC_XGB_COL = "PredictedRemoteWallClockTime_xgb"
SEC_LGB_COL = "PredictedRemoteWallClockTime_lgb"
BKT_XGB_COL = "pred_bucket_xgb"
BKT_LGB_COL = "pred_bucket_lgb"

# ========= Ͱ���ã����ӣ� =========
BUCKET_EDGES_MINUTES = [0, 30, 60, 180, 360, 720, float("inf")]
BUCKET_LABELS = ["<30m", "30-60m", "1-3h", "3-6h", "6-12h", ">=12h"]

def _minutes_to_seconds_edges(mins):
    return [float("inf") if m == float("inf") else int(m*60) for m in mins]

BINS = np.array(_minutes_to_seconds_edges(BUCKET_EDGES_MINUTES), dtype=float)
INTERNAL_BINS = BINS[1:-1] if np.isinf(BINS[-1]) else BINS[1:]
LOWER_BOUNDS = BINS[:-1]
UPPER_BOUNDS = BINS[1:]
N_BUCKETS = len(LOWER_BOUNDS)
BUCKET_IDS = list(range(N_BUCKETS))

def bucketize_seconds(arr: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(arr, float), INTERNAL_BINS, right=False)

# ========= ��Ȩ�������� =========
W_GRID = np.linspace(0.0, 1.0, 11)  # xgb Ȩ��
LAMBDA_BUCKET = 0.20                # RMSE + Ͱƫ�Ȩ

# ========= ABCD �������� =========
# A: no shift, no clip
# B: no shift, clip
# C: shift, no clip
# D: shift, clip
SHIFT_SEC = 600.0  # ͳһ���Ƶİ�ȫԣ�����룩���ɰ���Ҫ����
STRATEGIES = ["A_noShift_noClip", "B_noShift_clip", "C_shift_noClip", "D_shift_clip"]

def _bucket_bounds(b: int) -> Tuple[float, float]:
    lo = LOWER_BOUNDS[b]
    hi = UPPER_BOUNDS[b]
    return (float(lo), float(hi))

def clip_to_bucket_with_given_labels(preds: np.ndarray, buckets: np.ndarray) -> np.ndarray:
#    �� preds ��ֵ�ü�������Ͱ��ǩ�������ڣ����� B/D ��Ͱ�ü���
    preds = np.asarray(preds, float)
    b = np.asarray(buckets, int)
    out = preds.copy()
    for i, bi in enumerate(b):
        lo, hi = _bucket_bounds(int(bi))
        if np.isfinite(hi):
            out[i] = np.clip(out[i], lo, hi - 1e-6)
        else:
            out[i] = max(out[i], lo)
    return out

def score_rmse_plus_bucket(y_true_sec, y_pred_sec, tb, pb) -> float:
#  ���� weighted_search ��Ŀ�꣺RMSE + Ͱƫ�
    rmse = float(np.sqrt(np.mean((y_true_sec - y_pred_sec) ** 2)))
    bpen = float(np.mean(np.abs(pb - tb)))
    return rmse + LAMBDA_BUCKET * bpen

def direction_counts(tb: np.ndarray, pb: np.ndarray) -> Dict[str, int]:
    dev = pb - tb
    return {
        "Hit":   int(np.sum(dev == 0)),
        "Under": int(np.sum(dev < 0)),
        "Over":  int(np.sum(dev > 0)),
        "Total": int(len(dev)),
    }

# ========= ������ =========
def main():
    df = pd.read_csv(INPUT_CSV)
    cols = set(df.columns)
    print(f"[INFO] Loaded {len(df)} rows.")

    # true_bucket
    if TRUE_BKT not in cols:
        if Y_TRUE_COL in cols:
            df[TRUE_BKT] = bucketize_seconds(df[Y_TRUE_COL].to_numpy(float))
            print("[INFO] true_bucket reconstructed from y_true.")
        else:
            raise ValueError("Need either true_bucket or RemoteWallClockTime for evaluation.")

    # ��Ҫ����
    needed = [SEC_XGB_COL, SEC_LGB_COL, BKT_XGB_COL, BKT_LGB_COL, TRUE_BKT]
    for c in needed:
        if c not in cols:
            raise ValueError(f"Missing column: {c}")

    # ���� NA/Inf
    key_cols = [Y_TRUE_COL] if Y_TRUE_COL in cols else []
    key_cols += [SEC_XGB_COL, SEC_LGB_COL, BKT_XGB_COL, BKT_LGB_COL, TRUE_BKT]
    keep = df[key_cols].replace([np.inf, -np.inf], np.nan).dropna().index
    if len(keep) < len(df):
        print(f"[WARN] Dropped {len(df)-len(keep)} rows with NA/Inf in key columns.")
    df = df.loc[keep].reset_index(drop=True)

    # ��������
    y_true = df[Y_TRUE_COL].to_numpy(float) if Y_TRUE_COL in cols else None
    tb     = df[TRUE_BKT].to_numpy(int)
    has_pid = (PID_COL in df.columns)

    sec_x = df[SEC_XGB_COL].to_numpy(float)
    pb_x  = df[BKT_XGB_COL].to_numpy(int)
    sec_l = df[SEC_LGB_COL].to_numpy(float)
    pb_l  = df[BKT_LGB_COL].to_numpy(int)

    # ----- 1) xgb_only / 2) lgb_only -----
    methods: Dict[str, Dict[str, np.ndarray]] = {}
    methods["xgb_only"] = {"sec": sec_x, "pb": pb_x}
    methods["lgb_only"] = {"sec": sec_l, "pb": pb_l}

    # ----- 3) best_of_two���� PID �� Accuracy ���ţ� -----
    pb_best = np.zeros_like(tb)
    sec_best = np.zeros_like(sec_x, dtype=float)
    if has_pid:
        for pid, g in df.groupby(PID_COL):
            idx = g.index.values
            acc_x = float(np.mean(pb_x[idx] == tb[idx]))
            acc_l = float(np.mean(pb_l[idx] == tb[idx]))
            if acc_x >= acc_l:
                pb_best[idx]  = pb_x[idx]
                sec_best[idx] = sec_x[idx]
            else:
                pb_best[idx]  = pb_l[idx]
                sec_best[idx] = sec_l[idx]
    else:
        # �� PID ʱ�������� Accuracy ����
        if float(np.mean(pb_x == tb)) >= float(np.mean(pb_l == tb)):
            pb_best, sec_best = pb_x, sec_x
        else:
            pb_best, sec_best = pb_l, sec_l
    methods["best_of_two"] = {"sec": sec_best, "pb": pb_best}

    # ----- 4) weighted_search�������������Լ�Ȩ��Ŀ�꣺RMSE+Ͱ�ͷ��� -----
    best_w, best_score = None, np.inf
    for w in W_GRID:
        sec_w = w * sec_x + (1.0 - w) * sec_l
        pb_w  = bucketize_seconds(sec_w)
        s = score_rmse_plus_bucket(y_true, sec_w, tb, pb_w) if y_true is not None else np.mean(np.abs(pb_w - tb))
        if s < best_score:
            best_score = s; best_w = float(w)
    sec_ws = best_w * sec_x + (1.0 - best_w) * sec_l
    pb_ws  = bucketize_seconds(sec_ws)
    print(f"[OK] weighted_search: w_xgb={best_w:.2f}, score={best_score:.6f}")
    methods["weighted_search"] = {"sec": sec_ws, "pb": pb_ws}

    # ===== ABCD ����ʵ�飺Hit / Under / Over / Total / Hit Rate(%) =====
    summary_rows = []

    for name, pred in methods.items():
        base_sec = pred["sec"]
        # �ԡ�δ shift δ clip����Ԥ��Ͱ����Ϊ B/D �ġ���Ͱ������
        base_pb  = bucketize_seconds(base_sec)

        # A: no shift, no clip
        sec_A = base_sec
        pb_A  = bucketize_seconds(sec_A)

        # B: no shift, clip���� A ��Ͱ�������� shift ������ͬ A��
        sec_B = clip_to_bucket_with_given_labels(sec_A, base_pb)
        pb_B  = bucketize_seconds(sec_B)

        # C: shift, no clip��ͳһ���ƣ�
        sec_C = base_sec + SHIFT_SEC
        pb_C  = bucketize_seconds(sec_C)

        # D: shift, clip�������ƣ��ٰ�������ǰ��Ͱ������Ͱ�ü�����ֹ��Ͱ��
        sec_D = clip_to_bucket_with_given_labels(sec_C, base_pb)
        pb_D  = bucketize_seconds(sec_D)

        strat_map = {
            "A_noShift_noClip": (sec_A, pb_A),
            "B_noShift_clip":   (sec_B, pb_B),
            "C_shift_noClip":   (sec_C, pb_C),
            "D_shift_clip":     (sec_D, pb_D),
        }

        for strat, (_, pbX) in strat_map.items():
            cnts = direction_counts(tb, pbX)
            hit_rate = (cnts["Hit"] / cnts["Total"] * 100.0) if cnts["Total"] > 0 else np.nan
            summary_rows.append({
                "method": name,
                "strategy": strat,
                "Hit": cnts["Hit"],
                "Under": cnts["Under"],
                "Over": cnts["Over"],
                "Total": cnts["Total"],
                "Hit Rate (%)": round(hit_rate, 4),
            })

    # ���� CSV
    summary_df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(OUTPUT_DIR, "strategy_summary_all.csv")
    summary_df.to_csv(out_csv, index=False)

    # ��¼��Ȩ����������Ȩ�أ�����
    with open(os.path.join(OUTPUT_DIR, "weighted_search_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"weighted_search": {"w_xgb": float(best_w),
                                       "objective": "rmse+bucket_pen",
                                       "lambda_bucket": LAMBDA_BUCKET}},
                  f, ensure_ascii=False, indent=2)

    print(f"[OK] ABCD strategy summaries saved to: {out_csv}")
    print(f"[OK] Output dir: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
