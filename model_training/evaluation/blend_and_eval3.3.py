# -*- coding: utf-8 -*-
# blend_and_eval3.2.py — Regression eval (v3.3)
# 读取已完成的预测 CSV（XGB/LGB），生成四种策略（xgb/lgb/best/weighted-search）的评估报表与图表
from __future__ import annotations
import os, json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= 路径配置（回归）=========
XGB_CSV = "/home/master/wzheng/projects/model_training/preds/predictions_v3.3_xgb.csv"
LGB_CSV = "/home/master/wzheng/projects/model_training/preds/predictions_v3.3_lgb.csv"
OUT_DIR = "/home/master/wzheng/projects/model_training/evaluation/v3.3_regression"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 字段名约定 =========
PID_COL   = "ProgramID_encoded"
Y_TRUE    = "RemoteWallClockTime"
Y_PRED    = "PredictedRemoteWallClockTime"  # 每份预测里都有这个列名

# ========= 桶（用于分类化评估 & 图表对齐可选）=========
BUCKET_EDGES_MINUTES = [0, 30, 60, 180, 360, 720, float("inf")]
def _m2s(m): return [float("inf") if x == float("inf") else int(x*60) for x in m]
_BINS = np.array(_m2s(BUCKET_EDGES_MINUTES), float)
_INTERNAL = _BINS[1:-1] if np.isinf(_BINS[-1]) else _BINS[1:]   # digitize 内部分界
LOWER = _BINS[:-1]; UPPER = _BINS[1:]
N_BUCKETS = len(LOWER)
BUCKET_LABELS = []
for i in range(N_BUCKETS):
    l = LOWER[i]; r = UPPER[i]
    def _fmt(sec):
        if np.isinf(sec): return "inf"
        m = sec/60.0
        if m < 60: return f"{int(m)}m"
        h = m/60.0
        return f"{int(h)}h"
    if np.isinf(r):
        BUCKET_LABELS.append(f"[{_fmt(l)}, +inf)")
    else:
        BUCKET_LABELS.append(f"[{_fmt(l)}, {_fmt(r)})")

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
    tb = _bucketize_sec(y); pb = _bucketize_sec(p)
    acc  = float(np.mean(pb == tb))
    acc1 = float(np.mean(np.abs(pb - tb) <= 1))
    return {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "Bias": bias, "R2": r2,
            "Accuracy": acc, "Accuracy+-1": acc1}

# ========= 工具 =========
def _read_pred_csv(path: str, model_tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert PID_COL in df.columns and Y_TRUE in df.columns and Y_PRED in df.columns, \
        f"CSV missing column: {PID_COL}/{Y_TRUE}/{Y_PRED} at {path}"
    out = df[[PID_COL, Y_TRUE, Y_PRED]].copy()
    out.rename(columns={Y_PRED: f"pred_{model_tag}"}, inplace=True)
    return out

def _merge_preds(xgb_df: pd.DataFrame, lgb_df: pd.DataFrame) -> pd.DataFrame:
    # 以行号对齐（两份预测应来自同一评估集），若你需要严格键对齐，可改为 keys merge
    if len(xgb_df) != len(lgb_df):
        # 容错：外连接后丢 NA
        df = pd.merge(xgb_df.reset_index().rename(columns={"index":"rid"}),
                      lgb_df.reset_index().rename(columns={"index":"rid"}),
                      on=["rid", PID_COL, Y_TRUE], how="outer")
        df = df.dropna(subset=[Y_TRUE]).reset_index(drop=True)
        return df.drop(columns=["rid"])
    else:
        df = xgb_df.copy()
        df["pred_lgb"] = lgb_df["pred_lgb"].values
        return df

def _weighted_search(y, px, pl) -> Tuple[float, np.ndarray]:
    # 在 {0,0.1,...,1} 上搜索使 RMSE 最小（也可扩展目标）
    grid = np.linspace(0.0, 1.0, 11)
    best_w, best_rmse, best_pred = 0.0, float("inf"), None
    for w in grid:
        p = w*px + (1.0-w)*pl
        rmse = np.sqrt(mean_squared_error(y, p))
        if rmse < best_rmse:
            best_rmse, best_w, best_pred = rmse, float(w), p
    return best_w, best_pred

def _ensure_dir(p): 
    d = os.path.dirname(p); 
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ========= 新增：每桶命中率统计 & 图 =========
def per_bucket_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    tb = _bucketize_sec(y_true)
    pb = _bucketize_sec(y_pred)
    rows = []
    for b in range(N_BUCKETS):
        mask = (tb == b)
        n = int(mask.sum())
        if n == 0:
            hit = np.nan
        else:
            hit = float(np.mean(pb[mask] == b))
        rows.append({"BucketID": b, "BucketLabel": BUCKET_LABELS[b], "Count": n, "HitRate": hit})
    df = pd.DataFrame(rows)
    overall = float(np.mean(pb == tb)) if len(y_true) > 0 else np.nan
    df.attrs["overall_accuracy"] = overall
    return df

def plot_bucket_accuracy_bar(df_acc: pd.DataFrame, title: str, save_path: str):
    plt.figure()
    ax = df_acc.set_index("BucketLabel")["HitRate"].plot(kind="bar")
    # 在柱子上标注
    vals = df_acc["HitRate"].values
    for i, v in enumerate(vals):
        if pd.notna(v):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1)
    plt.ylabel("Hit Rate")
    plt.title(title)
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close()

def plot_bucket_accuracy_lines(acc_dict: Dict[str, pd.DataFrame], save_path: str):
    # acc_dict: {variant: df_acc}
    plt.figure()
    xs = list(range(N_BUCKETS))
    labels = [r["BucketLabel"] for _, r in acc_dict[next(iter(acc_dict))].iterrows()]
    for name, df_acc in acc_dict.items():
        ys = df_acc["HitRate"].values
        plt.plot(xs, ys, marker="o", label=name)
    plt.xticks(xs, labels, rotation=0)
    plt.ylim(0, 1)
    plt.xlabel("Bucket")
    plt.ylabel("Hit Rate")
    plt.title("Per-bucket Hit Rate (All Variants)")
    plt.legend()
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.close()

# ========= 主流程 =========
def main():
    df_x = _read_pred_csv(XGB_CSV, "xgb")
    df_l = _read_pred_csv(LGB_CSV, "lgb")
    df   = _merge_preds(df_x, df_l)

    # 四种策略：xgb, lgb, best(逐PID/含Others), weighted-search(逐PID/含Others)
    variants = {}
    # base
    variants["xgb"] = df["pred_xgb"].to_numpy(float)
    variants["lgb"] = df["pred_lgb"].to_numpy(float)

    # choose-best per PID（评估集上挑 RMSE 更小的）
    pred_best = np.zeros(len(df), float)
    for pid, g in df.groupby(PID_COL):
        y  = g[Y_TRUE].to_numpy(float)
        px = g["pred_xgb"].to_numpy(float)
        pl = g["pred_lgb"].to_numpy(float)
        rmse_x = np.sqrt(mean_squared_error(y, px))
        rmse_l = np.sqrt(mean_squared_error(y, pl))
        use = px if rmse_x <= rmse_l else pl
        pred_best[g.index] = use
    variants["best"] = pred_best

    # weighted-search per PID（在 0..1 搜索最优 w）
    pred_ws = np.zeros(len(df), float)
    ws_by_pid = {}
    for pid, g in df.groupby(PID_COL):
        y  = g[Y_TRUE].to_numpy(float)
        px = g["pred_xgb"].to_numpy(float)
        pl = g["pred_lgb"].to_numpy(float)
        w, p = _weighted_search(y, px, pl)
        ws_by_pid[pid] = w
        pred_ws[g.index] = p
    variants["weighted"] = pred_ws
    pd.DataFrame([{"PID": pid, "weight_xgb": w, "weight_lgb": 1.0-w} for pid, w in ws_by_pid.items()])\
      .to_csv(os.path.join(OUT_DIR, "per_pid_weights.csv"), index=False)

    # 指标输出 & 图表
    summary_rows = []
    per_variant_bucket_acc = {}  # 用于合并折线图
    for name, pred in variants.items():
        mets = overall_metrics(df[Y_TRUE].values, pred)
        row = {"variant": name, **mets}
        summary_rows.append(row)

        # 每 PID 指标
        per_pid = []
        for pid, g in df.groupby(PID_COL):
            mets_pid = overall_metrics(g[Y_TRUE].values, pred[g.index])
            per_pid.append({"ProgramID_encoded": pid, **mets_pid, "n": len(g)})
        pd.DataFrame(per_pid).to_csv(os.path.join(OUT_DIR, f"per_pid_{name}.csv"), index=False)

        # 保存预测明细
        dsave = df[[PID_COL, Y_TRUE]].copy()
        dsave[f"pred_{name}"] = pred
        dsave.to_csv(os.path.join(OUT_DIR, f"pred_detail_{name}.csv"), index=False)

        # 图：Error vs True (log-log)
        abs_err = np.abs(pred - df[Y_TRUE].to_numpy(float))
        plt.figure()
        plt.scatter(df[Y_TRUE].to_numpy(float)+1.0, abs_err+1.0, s=3)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("True Runtime (sec, log)"); plt.ylabel("Absolute Error (sec, log)")
        plt.title(f"[v3.3] Error vs True (log-log) - {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"error_vs_true_loglog_{name}.png"))
        plt.close()

        # ===== 新增：每桶命中率统计 & 图 =====
        acc_df = per_bucket_accuracy(df[Y_TRUE].to_numpy(float), pred)
        acc_df.to_csv(os.path.join(OUT_DIR, f"per_bucket_accuracy_{name}.csv"), index=False)
        # 在文件名里也附带一下整体命中率，阅读更直观
        overall_acc = acc_df.attrs.get("overall_accuracy", np.nan)
        plot_bucket_accuracy_bar(
            acc_df,
            title=f"[v3.3] Per-bucket Hit Rate - {name} (overall={overall_acc:.3f})",
            save_path=os.path.join(OUT_DIR, f"per_bucket_accuracy_bar_{name}.png")
        )
        per_variant_bucket_acc[name] = acc_df[["BucketID","BucketLabel","HitRate"]].copy()

    # 汇总
    comp_df = pd.DataFrame(summary_rows)
    comp_df.to_csv(os.path.join(OUT_DIR, "overall_summary.csv"), index=False)

    # 对比条形图（RMSE/MAE/MAPE/Accuracy/Accuracy+-1）
    for k in ["RMSE", "MAE", "MAPE(%)", "Accuracy", "Accuracy+-1"]:
        plt.figure()
        comp_df.plot(x="variant", y=k, kind="bar", legend=False)
        plt.ylabel(k); plt.title(f"[v3.3] Variant Comparison - {k}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"compare_{k}.png"))
        plt.close()

    # ===== 新增：四种策略的每桶命中率合并折线图 =====
    # 合并为表：BucketID, BucketLabel, xgb, lgb, best, weighted
    merged = None
    for name, df_acc in per_variant_bucket_acc.items():
        df_acc = df_acc.rename(columns={"HitRate": name})
        merged = df_acc if merged is None else pd.merge(
            merged, df_acc, on=["BucketID","BucketLabel"], how="outer"
        )
    merged = merged.sort_values("BucketID").reset_index(drop=True)
    merged.to_csv(os.path.join(OUT_DIR, "per_bucket_accuracy_all_variants.csv"), index=False)

    # 折线图
    plot_bucket_accuracy_lines(per_variant_bucket_acc, save_path=os.path.join(OUT_DIR, "per_bucket_accuracy_all_variants.png"))

    print(f"[OK] Regression eval done. Results -> {OUT_DIR}")

if __name__ == "__main__":
    main()
