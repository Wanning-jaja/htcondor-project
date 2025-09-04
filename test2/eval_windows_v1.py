# -*- coding: utf-8 -*-
# 简化版评估（窗口对比 + weighted 融合 + 总表汇总）
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= 路径配置（按你现在的文件名与目录）=========
PRED_PATHS = {
    # XGB
    "90_xgb":   "/home/master/wzheng/projects/test2/preds/predictions_v3_90_xgb.csv",
    "182_xgb":  "/home/master/wzheng/projects/test2/preds/predictions_v3_182_xgb.csv",
    "full_xgb": "/home/master/wzheng/projects/model_training/preds/predictions_v3.3_xgb.csv",
    # LGB
    "90_lgb":   "/home/master/wzheng/projects/test2/preds/predictions_v3_90_lgb.csv",
    "182_lgb":  "/home/master/wzheng/projects/test2/preds/predictions_v3_182_lgb.csv",
    "full_lgb": "/home/master/wzheng/projects/model_training/preds/predictions_v3.3_lgb.csv",
}

OUT_DIR = "/home/master/wzheng/projects/test2/evaluation/windows_v2"  # 改一个新目录避免覆盖
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 字段名 =========
PID_COL   = "ProgramID_encoded"
Y_TRUE    = "RemoteWallClockTime"
Y_PRED    = "PredictedRemoteWallClockTime"

# ========= 桶（保持与原评估一致）=========
BUCKET_EDGES_MINUTES = [0, 30, 60, 180, 360, 720, float("inf")]
def _m2s(m): return [float("inf") if x == float("inf") else int(x*60) for x in m]
_BINS = np.array(_m2s(BUCKET_EDGES_MINUTES), float)
_INTERNAL = _BINS[1:-1] if np.isinf(_BINS[-1]) else _BINS[1:]
LOWER = _BINS[:-1]; UPPER = _BINS[1:]
N_BUCKETS = len(LOWER)

BUCKET_LABELS = []
def _fmt_sec(sec):
    if np.isinf(sec): return "inf"
    m = sec / 60.0
    if m < 60: return f"{int(m)}m"
    h = m / 60.0
    return f"{int(h)}h"

for i in range(N_BUCKETS):
    l = LOWER[i]; r = UPPER[i]
    if np.isinf(r):
        BUCKET_LABELS.append(f"[{_fmt_sec(l)}, +inf)")
    else:
        BUCKET_LABELS.append(f"[{_fmt_sec(l)}, {_fmt_sec(r)})")

def _bucketize_sec(a: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(a, float), _INTERNAL, right=False)

# ========= 指标 =========
def safe_mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = (np.abs(y) > 1e-9) & np.isfinite(p)
    return float(np.mean(np.abs((p[m]-y[m])/y[m]))*100.0) if np.any(m) else np.nan

def overall_metrics(y, p) -> Dict[str, float]:
    y = np.asarray(y, float); p = np.asarray(p, float)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae  = float(mean_absolute_error(y, p))
    mape = safe_mape(y, p)
    bias = float(np.mean(p - y))
    try: r2 = float(r2_score(y, p))
    except: r2 = np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "Bias": bias, "R2": r2}

def compute_bucket_accuracy(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y, float); p = np.asarray(p, float)
    true_b = _bucketize_sec(y); pred_b = _bucketize_sec(p)
    rows = []
    for b in range(N_BUCKETS):
        m = (true_b == b)
        n = int(np.sum(m))
        if n == 0:
            rows.append({"BucketID": b, "BucketLabel": BUCKET_LABELS[b], "Count": 0, "Hits": 0, "HitRate": np.nan})
        else:
            hits = int(np.sum(pred_b[m] == b))
            rows.append({"BucketID": b, "BucketLabel": BUCKET_LABELS[b], "Count": n, "Hits": hits, "HitRate": hits / n})
    return pd.DataFrame(rows)

# ========= IO =========
def read_pred_one(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for c in (PID_COL, Y_TRUE, Y_PRED):
        if c not in df.columns:
            raise ValueError(f"{path} missing column: {c}")
    df = df[[PID_COL, Y_TRUE, Y_PRED]].copy()
    df[Y_TRUE] = pd.to_numeric(df[Y_TRUE], errors="coerce")
    df[Y_PRED] = pd.to_numeric(df[Y_PRED], errors="coerce")
    df = df[np.isfinite(df[Y_TRUE]) & np.isfinite(df[Y_PRED])]
    return df.reset_index(drop=True)

def read_pair(window_tag: str) -> pd.DataFrame:
#    读取同一窗口的 XGB/LGB 预测，并按行对齐成一张表：y, pred_xgb, pred_lgb
    xgb = read_pred_one(PRED_PATHS[f"{window_tag}_xgb"]).rename(columns={Y_PRED: "pred_xgb"})
    lgb = read_pred_one(PRED_PATHS[f"{window_tag}_lgb"]).rename(columns={Y_PRED: "pred_lgb"})
    if len(xgb) != len(lgb):
        # 容错外连接 + 丢 NA（保留共有样本）
        df = pd.merge(xgb.reset_index().rename(columns={"index":"rid"}),
                      lgb.reset_index().rename(columns={"index":"rid"}),
                      on=["rid", PID_COL, Y_TRUE], how="outer")
        df = df.dropna(subset=[Y_TRUE]).reset_index(drop=True)
        return df.drop(columns=["rid"])
    else:
        df = xgb.copy()
        df["pred_lgb"] = lgb["pred_lgb"].values
        return df

# ========= weighted（逐 PID 搜索最优权重，参考你上传的 v3.3 脚本）=========
def _weighted_search(y, px, pl) -> Tuple[float, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 11)  # {0,0.1,...,1}
    best_w, best_rmse, best_pred = 0.0, float("inf"), None
    for w in grid:
        p = w*px + (1.0-w)*pl
        rmse = np.sqrt(mean_squared_error(y, p))
        if rmse < best_rmse:
            best_rmse, best_w, best_pred = rmse, float(w), p
    return best_w, best_pred

def build_weighted_pred_per_window(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
#    返回 weighted 预测向量 & PID 权重表
    pred_ws = np.zeros(len(df), float)
    rows = []
    for pid, g in df.groupby(PID_COL):
        y  = g[Y_TRUE].to_numpy(float)
        px = g["pred_xgb"].to_numpy(float)
        pl = g["pred_lgb"].to_numpy(float)
        w, p = _weighted_search(y, px, pl)
        pred_ws[g.index] = p
        rows.append({"ProgramID_encoded": pid, "weight_xgb": w, "weight_lgb": 1.0 - w, "n": len(g)})
    wdf = pd.DataFrame(rows).sort_values("ProgramID_encoded").reset_index(drop=True)
    return pred_ws, wdf

# ========= 作图 =========
def plot_lines(acc_dict: Dict[str, pd.DataFrame], title: str, save_path: str):
    plt.figure()
    xs = list(range(N_BUCKETS))
    labels = [r["BucketLabel"] for _, r in next(iter(acc_dict.values())).iterrows()]
    for name, df_acc in acc_dict.items():
        ys = df_acc["HitRate"].values
        plt.plot(xs, ys, marker="o", label=name)
    plt.xticks(xs, labels, rotation=0)
    plt.ylim(0, 1)
    plt.xlabel("Bucket")
    plt.ylabel("Hit Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ========= 主流程 =========
def main():
    # ---- 原始逐文件评估（保留你已有 XGB/LGB 单模型输出）----
    summaries = {}
    bucket_accs = {}
    for tag, path in PRED_PATHS.items():
        df = read_pred_one(path)
        mets = overall_metrics(df[Y_TRUE].values, df[Y_PRED].values)
        acc  = compute_bucket_accuracy(df[Y_TRUE].values, df[Y_PRED].values)
        overall_hit = float(np.nansum(acc["Hits"])) / float(np.nansum(acc["Count"])) if np.nansum(acc["Count"]) > 0 else np.nan
        summary = pd.DataFrame([{ "Tag": tag, "N": len(df), **mets, "HitOverall": overall_hit }])
        summary.to_csv(os.path.join(OUT_DIR, f"summary_{tag}.csv"), index=False)
        acc.to_csv(os.path.join(OUT_DIR, f"bucket_acc_{tag}.csv"), index=False)
        summaries[tag] = summary
        bucket_accs[tag] = acc

    # ---- 你的需求 ①：对比不同训练窗口的 weighted（xgb+lgb）分桶命中率 ----
    window_tags = ["90", "182", "full"]
    weighted_acc_by_window: Dict[str, pd.DataFrame] = {}
    merged_weighted_acc = None

    for win in window_tags:
        df_pair = read_pair(win)                        # y, pred_xgb, pred_lgb
        pred_w, wdf = build_weighted_pred_per_window(df_pair)
        # 保存每窗的 PID 权重
        wdf.to_csv(os.path.join(OUT_DIR, f"per_pid_weights_{win}.csv"), index=False)

        # 分桶命中率
        acc_df = compute_bucket_accuracy(df_pair[Y_TRUE].to_numpy(float), pred_w)
        acc_df.rename(columns={"HitRate": f"HitRate_{win}"}, inplace=True)
        acc_df.to_csv(os.path.join(OUT_DIR, f"per_bucket_weighted_{win}.csv"), index=False)

        # 用于合并与画折线
        weighted_acc_by_window[win] = acc_df.rename(columns={f"HitRate_{win}": "HitRate"})

        # 合表（BucketID, BucketLabel, HitRate_90, HitRate_182, HitRate_full）
        merged_weighted_acc = acc_df if merged_weighted_acc is None else \
            pd.merge(merged_weighted_acc, acc_df[["BucketID","BucketLabel",f"HitRate_{win}"]],
                     on=["BucketID","BucketLabel"], how="outer")

    merged_weighted_acc = merged_weighted_acc.sort_values("BucketID").reset_index(drop=True)
    merged_weighted_acc.to_csv(os.path.join(OUT_DIR, "per_bucket_weighted_all_windows.csv"), index=False)

    # 折线图：weighted 三条线（90/182/full）
    plot_lines(
        weighted_acc_by_window,
        title="Per-bucket Hit Rate (Weighted: 90 / 182 / full)",
        save_path=os.path.join(OUT_DIR, "per_bucket_weighted_3lines.png")
    )

    # ---- 你的需求 ②：合并 6 份 summary 成 overall 总表 ----
    # 文件名与你当前输出一致：summary_182_lgb.csv 等
    need_files = [
        "summary_182_lgb.csv", "summary_182_xgb.csv",
        "summary_90_lgb.csv",  "summary_90_xgb.csv",
        "summary_full_lgb.csv","summary_full_xgb.csv",
    ]
    dfs = []
    for fn in need_files:
        fp = os.path.join(OUT_DIR, fn)
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            # 增加一个来源字段（从文件名推断）
            src = fn.replace("summary_", "").replace(".csv", "")
            df.insert(0, "Source", src)
            dfs.append(df)
    if len(dfs) > 0:
        overall = pd.concat(dfs, ignore_index=True)
        overall.to_csv(os.path.join(OUT_DIR, "overall_summary_all_6.csv"), index=False)
    else:
        print("No summary_*.csv files found. Verify that OUT_DIR is correct.")

    # ----（可选）保留原本对比图表 ----
    xgb_tags = ["90_xgb", "182_xgb", "full_xgb"]
    lgb_tags = ["90_lgb", "182_lgb", "full_lgb"]
    def _concat(tags):
        ok = [summaries[t] for t in tags if t in summaries]
        return pd.concat(ok, ignore_index=True) if ok else pd.DataFrame()
    xgb_sum = _concat(xgb_tags); lgb_sum = _concat(lgb_tags)
    if not xgb_sum.empty: xgb_sum.to_csv(os.path.join(OUT_DIR, "compare_xgb_summary.csv"), index=False)
    if not lgb_sum.empty: lgb_sum.to_csv(os.path.join(OUT_DIR, "compare_lgb_summary.csv"), index=False)

    # 画单模型折线（原逻辑保留）
    def _pick(d, keys): return {t: d[t] for t in keys if t in d}
    plot_lines(_pick(bucket_accs, xgb_tags),
               title="Per-bucket Hit Rate (XGB: 90/182/full)",
               save_path=os.path.join(OUT_DIR, "per_bucket_xgb_3lines.png"))
    plot_lines(_pick(bucket_accs, lgb_tags),
               title="Per-bucket Hit Rate (LGB: 90/182/full)",
               save_path=os.path.join(OUT_DIR, "per_bucket_lgb_3lines.png"))

    print("[OK] Eval done ->", OUT_DIR)
    if not xgb_sum.empty: print("XGB summary:\n", xgb_sum)
    if not lgb_sum.empty: print("LGB summary:\n", lgb_sum)

if __name__ == "__main__":
    main()
