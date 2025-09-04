# -*- coding: utf-8 -*-
# evaluate_5.4.py — 真分类评估 + 内置加权搜索 + 四策略每桶命中率输出与合并折线
from __future__ import annotations
import os, json, warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ========= 路径 =========
INPUT_CSV  = "/home/master/wzheng/projects/model_training/preds/v5.4_cls_predictions.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/evaluation/v5.4_cls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= 字段 =========
Y_TRUE_COL  = "RemoteWallClockTime"
PID_COL     = "ProgramID_encoded"
TRUE_BKT    = "true_bucket"

# 预测列（至少要有 XGB/LGB 的“秒”列与对应桶列）
SEC_XGB_COL = "PredictedRemoteWallClockTime_xgb"
SEC_LGB_COL = "PredictedRemoteWallClockTime_lgb"
BKT_XGB_COL = "pred_bucket_xgb"
BKT_LGB_COL = "pred_bucket_lgb"

# *_cs（可选）
SEC_XGB_CS_COL = "PredictedRemoteWallClockTime_xgb_cs"
SEC_LGB_CS_COL = "PredictedRemoteWallClockTime_lgb_cs"
BKT_XGB_CS_COL = "pred_bucket_xgb_cs"
BKT_LGB_CS_COL = "pred_bucket_lgb_cs"

# ========= 桶配置（分钟） =========
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

def bucket_mid_seconds(b: int) -> float:
    lo, hi = LOWER_BOUNDS[b], UPPER_BOUNDS[b]
    return (lo + (hi if np.isfinite(hi) else lo*2.0)) / 2.0

# ========= 加权搜索参数 =========
W_GRID = np.linspace(0.0, 1.0, 11)  # xgb 权重
LAMBDA_BUCKET = 0.20                # RMSE + 桶偏差罚权
LAM_UNDER = 0.6                     # 方向代价：低估
LAM_OVER  = 0.2                     # 方向代价：高估

# ========= 指标 =========
def safe_mape(y, p) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(y) & (np.abs(y) > 1e-9) & np.isfinite(p)
    return float(np.mean(np.abs((p[m] - y[m]) / y[m])) * 100.0) if np.any(m) else np.nan

def overall_metrics(y_true: np.ndarray, y_pred_sec: np.ndarray,
                    tb: np.ndarray, pb: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_sec)))
    mae  = float(mean_absolute_error(y_true, y_pred_sec))
    mape = safe_mape(y_true, y_pred_sec)
    bias = float(np.mean(y_pred_sec - y_true))
    try: r2 = float(r2_score(y_true, y_pred_sec))
    except Exception: r2 = np.nan
    acc  = float(np.mean(pb == tb))
    acc1 = float(np.mean(np.abs(pb - tb) <= 1))
    under= float(np.mean(pb < tb))
    over = float(np.mean(pb > tb))
    hit  = float(np.mean(pb == tb))
    return {"rows": len(y_true), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape,
            "Bias": bias, "R2": r2, "Accuracy": acc, "Accuracy+-1": acc1,
            "under_rate": under, "over_rate": over, "hit_rate": hit}

def direction_cost(dev: np.ndarray, lam_under=LAM_UNDER, lam_over=LAM_OVER, scale=1000.0) -> float:
    dev = np.asarray(dev, dtype=float)
    under = np.clip(-dev, 0, None)
    over  = np.clip( dev, 0, None)
    return float(np.mean(lam_under * under + lam_over * over) * scale)

def score_rmse_plus_bucket(y_true_sec, y_pred_sec, tb, pb) -> float:
    rmse = float(np.sqrt(mean_squared_error(y_true_sec, y_pred_sec)))
    bpen = float(np.mean(np.abs(pb - tb)))
    return rmse + LAMBDA_BUCKET * bpen

def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d): os.makedirs(d, exist_ok=True)

# ========= 画图 =========
def plot_confusion(cm: pd.DataFrame, title: str, outpath: str):
    plt.figure(figsize=(8,6))
    plt.imshow(cm.values, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(ticks=np.arange(N_BUCKETS), labels=BUCKET_LABELS[:N_BUCKETS], rotation=45)
    plt.yticks(ticks=np.arange(N_BUCKETS), labels=BUCKET_LABELS[:N_BUCKETS])
    plt.xlabel("pred_bucket"); plt.ylabel("true_bucket")
    plt.title(title)
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_bar(values: pd.Series, title: str, outpath: str, ylim: Tuple[float,float] | None = None):
    plt.figure()
    ax = values.plot(kind="bar")
    if ylim: plt.ylim(*ylim)
    ymax = 0.01 if ylim else 0.01*max(values.values) if len(values) else 0.01
    for i, v in enumerate(values.values):
        if pd.notna(v):
            ax.text(i, v + ymax, f"{v:.2f}" if v<=1 else str(int(v)),
                    ha="center", va="bottom", fontsize=9)
    plt.title(title); plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_scatter_loglog(x_true_sec: np.ndarray, abs_err_sec: np.ndarray, outpath: str):
    plt.figure()
    plt.scatter(x_true_sec + 1.0, abs_err_sec + 1.0, s=3)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("True Runtime (sec, log)"); plt.ylabel("Absolute Error (sec, log)")
    plt.title("Error vs True Runtime (log-log)")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def plot_rel_err_cdf(y_true_sec: np.ndarray, y_pred_sec: np.ndarray, outpath: str):
    rel_err = np.abs((y_pred_sec - y_true_sec) / np.clip(np.abs(y_true_sec), 1e-9, None))
    rel_err = np.clip(rel_err, 0, 10)
    xs = np.sort(rel_err); ys = np.arange(1, len(xs)+1) / len(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Relative Error (|p-y|/|y|)"); plt.ylabel("CDF")
    plt.title("CDF of Relative Error")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# ========= 新增：每桶命中率表 =========
def per_bucket_accuracy(tb: np.ndarray, pb: np.ndarray) -> pd.DataFrame:
    rows = []
    overall_acc = float(np.mean(pb == tb))
    for b in BUCKET_IDS:
        mask = (tb == b)
        cnt = int(np.sum(mask))
        if cnt == 0:
            rows.append({"BucketID": b, "BucketLabel": BUCKET_LABELS[b] if b < len(BUCKET_LABELS) else str(b),
                         "Count": 0, "HitRate": np.nan, "OverallAccuracy": overall_acc})
        else:
            hit = float(np.mean(pb[mask] == b))
            rows.append({"BucketID": b, "BucketLabel": BUCKET_LABELS[b] if b < len(BUCKET_LABELS) else str(b),
                         "Count": cnt, "HitRate": hit, "OverallAccuracy": overall_acc})
    return pd.DataFrame(rows)

# ========= 主流程 =========
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

    # 需要的基线列
    needed = [SEC_XGB_COL, SEC_LGB_COL, BKT_XGB_COL, BKT_LGB_COL, TRUE_BKT]
    for c in needed:
        if c not in cols:
            raise ValueError(f"Missing column: {c}")

    # 过滤 NA/Inf
    key_cols = [Y_TRUE_COL] if Y_TRUE_COL in cols else []
    key_cols += [SEC_XGB_COL, SEC_LGB_COL, BKT_XGB_COL, BKT_LGB_COL, TRUE_BKT]
    keep = df[key_cols].replace([np.inf, -np.inf], np.nan).dropna().index
    if len(keep) < len(df):
        print(f"[WARN] Dropped {len(df)-len(keep)} rows with NA/Inf in key columns.")
    df = df.loc[keep].reset_index(drop=True)

    # 基础对象
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

    # ----- 3) best_of_two（按 PID 的 Accuracy 择优） -----
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
        # 无 PID 时，按整体 Accuracy 择优
        if float(np.mean(pb_x == tb)) >= float(np.mean(pb_l == tb)):
            pb_best, sec_best = pb_x, sec_x
        else:
            pb_best, sec_best = pb_l, sec_l
    methods["best_of_two"] = {"sec": sec_best, "pb": pb_best}

    # ----- 4) weighted_search（在秒域做线性加权，目标：RMSE+桶惩罚） -----
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

    # ====== 指标与图 ======
    overall_rows = []
    per_bucket_hit_map = {}  # 用于合并折线图

    for name, pred in methods.items():
        sec = pred["sec"]; pb = pred["pb"]
        dev = pb - tb

        # overall
        ov = overall_metrics(y_true, sec, tb, pb) if y_true is not None else {}
        row = {"method": name, **ov,
               "avg_deviation(bucket)": float(np.mean(dev)),
               "dir_cost(scale=1000, l_under=%.2f,l_over=%.2f)" % (LAM_UNDER, LAM_OVER): direction_cost(dev)}
        overall_rows.append(row)

        # 混淆矩阵 + 热力图
        cm = pd.crosstab(pd.Series(tb, name="true"), pd.Series(pb, name="pred")).reindex(index=BUCKET_IDS, columns=BUCKET_IDS, fill_value=0)
        cm.to_csv(os.path.join(OUTPUT_DIR, f"confusion_matrix_counts_{name}.csv"))
        plot_confusion(cm, f"Confusion Matrix ({name})", os.path.join(OUTPUT_DIR, f"confusion_matrix_heatmap_{name}.png"))

        # 命中/低估/高估条形图
        dir_counts = pd.Series({"Hit": np.sum(dev==0), "Under": np.sum(dev<0), "Over": np.sum(dev>0)}).astype(int)
        dir_counts.to_csv(os.path.join(OUTPUT_DIR, f"direction_counts_{name}.csv"))
        plot_bar(dir_counts, f"Bucket Result (Hit/Under/Over) - {name}", os.path.join(OUTPUT_DIR, f"direction_counts_bar_{name}.png"))

        # 对齐回归风格图（若有 y_true 秒）
        if y_true is not None:
            abs_err = np.abs(sec - y_true)
            plot_scatter_loglog(y_true, abs_err, os.path.join(OUTPUT_DIR, f"error_vs_true_loglog_{name}.png"))
            plot_rel_err_cdf(y_true, sec, os.path.join(OUTPUT_DIR, f"relative_error_cdf_{name}.png"))

        # ===== 新增：每桶命中率与整体命中率（CSV + 直方图） =====
        pba = per_bucket_accuracy(tb, pb)
        pba.to_csv(os.path.join(OUTPUT_DIR, f"per_bucket_accuracy_{name}.csv"), index=False)
        # 单独直方图
        plot_bar(pba.set_index("BucketLabel")["HitRate"],
                 f"Per-bucket Hit Rate - {name}",
                 os.path.join(OUTPUT_DIR, f"per_bucket_accuracy_bar_{name}.png"),
                 ylim=(0,1))
        # 收集到合并折线
        per_bucket_hit_map[name] = pba.set_index("BucketLabel")["HitRate"]

    # 汇总 overall
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(os.path.join(OUTPUT_DIR, "overall_metrics.csv"), index=False)

    # ===== 新增：四策略每桶命中率合并折线图 =====
    # 只画这四类：xgb_only/lgb_only/best_of_two/weighted_search（若部分缺失则自动跳过）
    order = ["xgb_only", "lgb_only", "best_of_two", "weighted_search"]
    hit_df = pd.DataFrame({k: per_bucket_hit_map[k] for k in order if k in per_bucket_hit_map})
    if not hit_df.empty:
        plt.figure()
        for col in hit_df.columns:
            plt.plot(hit_df.index.tolist(), hit_df[col].values, marker="o", label=col)
        plt.ylim(0, 1)
        plt.xlabel("Bucket"); plt.ylabel("Hit Rate")
        plt.title("Per-bucket Hit Rate (All Variants)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "per_bucket_hit_rate_all_variants_line.png"))
        plt.close()
        # 同时落盘合并表
        hit_df.to_csv(os.path.join(OUTPUT_DIR, "per_bucket_hit_rate_all_variants.csv"))

    # 记录加权搜索的最优权重
    with open(os.path.join(OUTPUT_DIR, "weighted_search_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"weighted_search": {"w_xgb": float(best_w), "objective": "rmse+bucket_pen", "lambda_bucket": LAMBDA_BUCKET}},
                  f, ensure_ascii=False, indent=2)

    print(f"\nAll evaluation artifacts saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
