# -*- coding: utf-8 -*-

#ProgramID 取舍建议脚本（v5.1 结果分析）
#------------------------------------
#输入：
#  - LGB 报告目录：/home/master/wzheng/projects/model_training/reports/v5.1_lgb/
#      - v5.1_lgb_evaluation_summary.csv
#      - pid_{pid}_bucket_metrics.csv（若存在 others_bucket_metrics.csv 会自动跳过）
#  - XGB 报告目录：/home/master/wzheng/projects/model_training/reports/v5.1_xgb/
#      - v5.1_xgb_evaluation_summary.csv
#      - pid_{pid}_bucket_metrics.csv

#输出（写入 LGB 报告目录下，便于集中查看）：
#  - programid_selection_merged.csv      （汇总分析表：两模型指标对齐 + 决策）
#  - programid_selection_decisions.csv   （只含决策与关键指标）
#  - programid_selection_keep_list.json  （建议保留的 ProgramID 列表）
#  - programid_selection_watchlist.json  （建议观察的 ProgramID 列表）
#  - programid_selection_merge_list.json （建议合入 Others 的 ProgramID 列表）

#可选：阈值可通过命令行修改，见 argparse 说明。


from __future__ import annotations
import os
import json
import argparse
import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# 默认路径（可用命令行覆盖）
# -------------------------
DEFAULT_LGB_DIR = "/home/master/wzheng/projects/model_training/reports/v5.1_lgb"
DEFAULT_XGB_DIR = "/home/master/wzheng/projects/model_training/reports/v5.1_xgb"
LGB_SUMMARY = "v5.1_lgb_evaluation_summary.csv"
XGB_SUMMARY = "v5.1_xgb_evaluation_summary.csv"

# -------------------------
# 阈值（可用命令行覆盖）
# -------------------------
DEFAULT_MIN_TOTAL_SAMPLES = 300     # Train+Val 总量不足则合并 Others
DEFAULT_MIN_VAL_SAMPLES   = 50      # 验证集过少则判为不稳定
DEFAULT_ACC_STRICT_KEEP   = 0.60    # 严格 Accuracy 达标线（任一模型达标即可）
DEFAULT_ACC_RELAX_KEEP    = 0.85    # 宽松 Accuracy±1 达标线（任一模型达标即可）
DEFAULT_BUCKET_MIN_ACC    = 0.40    # 桶内最低 Accuracy 的要求（若多桶明显低于此线 -> 不稳定）
DEFAULT_BUCKET_MIN_COUNT  = 30      # 仅对样本数不少于该阈值的桶做“最低 Accuracy”检查
DEFAULT_MANY_BUCKETS      = 4       # “多桶”判定的下限（用于识别“多桶全都不佳”的可疑 PID）


# ========= 工具函数 =========

def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read: {path} -> {e}")
        return None


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or np.all(weights <= 0):
        return np.nan
    s = np.sum(weights)
    if s <= 0:
        return np.nan
    return float(np.sum(values * weights) / s)


def _collect_bucket_agg(report_dir: str, pid: str | int) -> Dict[str, float]:
    
#    汇总某 PID 的 bucket-level 指标（若文件缺失或无列则返回 NaN）。
#    计算内容：
#      - weighted_acc / weighted_acc_relax（按 Count 加权）
#      - worst_bucket_acc（只统计 Count>=阈值的桶）
#      - bucket_count（有效桶数）
#      - buckets_below_thresh（低于最低 Accuracy 阈值的桶个数）

    path = os.path.join(report_dir, f"pid_{pid}_bucket_metrics.csv")
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return dict(
            weighted_acc=np.nan,
            weighted_acc_relax=np.nan,
            worst_bucket_acc=np.nan,
            bucket_count=0,
            buckets_below_thresh=0,
        )

    # 兼容列名（Accuracy±1 可能被写成 'Accuracy±1'）
    acc_col = "Accuracy" if "Accuracy" in df.columns else None
    acc_relax_col = "Accuracy+-1" if "Accuracy+-1" in df.columns else None

    # 没有 Accuracy 列就返回 NaN（v5.1 已经加了；旧版本可能没有）
    if acc_col is None or "Count" not in df.columns:
        return dict(
            weighted_acc=np.nan,
            weighted_acc_relax=np.nan,
            worst_bucket_acc=np.nan,
            bucket_count=int(df["Bucket"].nunique()) if "Bucket" in df.columns else 0,
            buckets_below_thresh=0,
        )

    count = df["Count"].fillna(0).astype(float).values
    acc_vals = df[acc_col].astype(float).values

    w_acc = _weighted_mean(acc_vals, count)
    if acc_relax_col and acc_relax_col in df.columns:
        w_acc_relax = _weighted_mean(df[acc_relax_col].astype(float).values, count)
    else:
        w_acc_relax = np.nan

    # 仅统计样本数足够的桶，做“最低 Accuracy”与“低于阈值桶数”检查
    df_big = df[(df["Count"].fillna(0).astype(float) >= args.bucket_min_count)].copy() if 'args' in globals() else df.copy()
    if df_big.empty or acc_col not in df_big.columns:
        worst_acc = np.nan
        below_cnt = 0
    else:
        worst_acc = float(np.min(df_big[acc_col].astype(float).values))
        thr = args.bucket_min_acc if 'args' in globals() else DEFAULT_BUCKET_MIN_ACC
        below_cnt = int((df_big[acc_col].astype(float).values < thr).sum())

    return dict(
        weighted_acc=w_acc,
        weighted_acc_relax=w_acc_relax,
        worst_bucket_acc=worst_acc,
        bucket_count=int(df["Bucket"].nunique()) if "Bucket" in df.columns else 0,
        buckets_below_thresh=below_cnt,
    )


def _merge_two_summaries(df_lgb: pd.DataFrame, df_xgb: pd.DataFrame) -> pd.DataFrame:
    
#    左右合并（outer），保留双方 size 与指标；公共键 ProgramID_encoded。
#    自动兼容列缺失（Accuracy/Accuracy±1 若 sumary 缺失，后续用 bucket 聚合补）。
    
    key = "ProgramID_encoded"
    on_cols = [key]

    # 防止重复列名冲突
    suffixes = ("_lgb", "_xgb")
    merged = pd.merge(df_lgb, df_xgb, on=on_cols, how="outer", suffixes=suffixes)

    # 按需确保某些关键列存在
    for col in [
        "TrainSize_lgb", "ValSize_lgb", "RMSE_lgb", "MAE_lgb", "MAPE(%)_lgb", "R2_lgb",
        "TrainSize_xgb", "ValSize_xgb", "RMSE_xgb", "MAE_xgb", "MAPE(%)_xgb", "R2_xgb",
        "Accuracy_lgb", "Accuracy+-1_lgb", "Accuracy_xgb", "Accuracy+-1_xgb"
    ]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged


def _standardize_summary(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    
#    标准化 summary 列名，加后缀 _{tag}，并只保留关键信息。
#    期望输入列包含：ProgramID_encoded, TrainSize, ValSize, RMSE, MAE, MAPE(%), R2, Accuracy, Accuracy±1
    
    keep = [
        "ProgramID_encoded", "TrainSize", "ValSize",
        "RMSE", "MAE", "MAPE(%)", "R2",
        "Accuracy", "Accuracy+-1"
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    rename = {c: f"{c}_{tag}" for c in cols if c != "ProgramID_encoded"}
    return out.rename(columns=rename)


def _decide_row(row: pd.Series, cfg: Dict[str, float]) -> Tuple[str, str]:
    
#    根据阈值做决策。返回 (Decision, Reason)：
#      - Keep
#      - MergeToOthers
#      - Watchlist
    
    # 数据规模
    ts_lgb = row.get("TrainSize_lgb", np.nan)
    vs_lgb = row.get("ValSize_lgb", np.nan)
    ts_xgb = row.get("TrainSize_xgb", np.nan)
    vs_xgb = row.get("ValSize_xgb", np.nan)

    total_size = np.nansum([ts_lgb, vs_lgb, ts_xgb, vs_xgb])  # 近似规模（两个模型同数据期望类似）
    val_size   = np.nanmax([vs_lgb, vs_xgb])

    # 准确率（summary 或 bucket 加权补齐）
    acc_lgb = row.get("Accuracy_lgb_weighted", row.get("Accuracy_lgb", np.nan))
    acc_xgb = row.get("Accuracy_xgb_weighted", row.get("Accuracy_xgb", np.nan))
    acc1_lgb = row.get("Accuracy+-1_lgb_weighted", row.get("Accuracy+-1_lgb", np.nan))
    acc1_xgb = row.get("Accuracy+-1_xgb_weighted", row.get("Accuracy+-1_xgb", np.nan))

    # 桶稳定性
    worst_lgb = row.get("worst_bucket_acc_lgb", np.nan)
    worst_xgb = row.get("worst_bucket_acc_xgb", np.nan)
    below_lgb = row.get("buckets_below_thresh_lgb", 0)
    below_xgb = row.get("buckets_below_thresh_xgb", 0)
    bucket_cnt_lgb = row.get("bucket_count_lgb", 0)
    bucket_cnt_xgb = row.get("bucket_count_xgb", 0)

    # 规则 1：数据太少 → 合并 Others
    if (not np.isnan(total_size) and total_size < cfg["min_total_samples"]) or \
       (not np.isnan(val_size) and val_size < cfg["min_val_samples"]):
        return "MergeToOthers", "too_few_samples"

    # 规则 2：性能达标（任一模型满足严格或宽松阈值）
    any_strict_ok = (not np.isnan(acc_lgb) and acc_lgb >= cfg["acc_strict_keep"]) or \
                    (not np.isnan(acc_xgb) and acc_xgb >= cfg["acc_strict_keep"])

    any_relax_ok  = (not np.isnan(acc1_lgb) and acc1_lgb >= cfg["acc_relax_keep"]) or \
                    (not np.isnan(acc1_xgb) and acc1_xgb >= cfg["acc_relax_keep"])

    # 规则 3：桶稳定性检查（Count>=阈值的桶里，是否存在明显低于阈值的情况）
    both_bad_bucket = False
    lgb_bad = (not np.isnan(worst_lgb)) and (worst_lgb < cfg["bucket_min_acc"]) and (below_lgb >= 1)
    xgb_bad = (not np.isnan(worst_xgb)) and (worst_xgb < cfg["bucket_min_acc"]) and (below_xgb >= 1)
    if lgb_bad and xgb_bad:
        both_bad_bucket = True

    # 规则 4：多桶且全面不佳（可疑 PID 划分）
    many_bucket = (bucket_cnt_lgb >= cfg["many_buckets"]) or (bucket_cnt_xgb >= cfg["many_buckets"])
    all_low = ( (np.isnan(acc_lgb) or acc_lgb < cfg["acc_strict_keep"]) and
                (np.isnan(acc_xgb) or acc_xgb < cfg["acc_strict_keep"]) and
                (np.isnan(acc1_lgb) or acc1_lgb < cfg["acc_relax_keep"]) and
                (np.isnan(acc1_xgb) or acc1_xgb < cfg["acc_relax_keep"]) )
    if many_bucket and all_low:
        return "MergeToOthers", "multi_bucket_consistently_low_accuracy"

    # 达标则优先 Keep；但若双模型都出现“桶不稳”，降级为 Watchlist
    if any_strict_ok or any_relax_ok:
        if both_bad_bucket:
            return "Watchlist", "accuracy_ok_but_bucket_unstable"
        return "Keep", "accuracy_threshold_met"

    # 双模型都不达标 → 合并 Others
    if not any_strict_ok and not any_relax_ok:
        if both_bad_bucket:
            return "MergeToOthers", "low_accuracy_and_bucket_unstable"
        return "MergeToOthers", "low_accuracy"

    # 其余情况 → 观察
    return "Watchlist", "borderline"


def main(args):
    # 1) 读取 summary
    lgb_sum_path = os.path.join(args.lgb_dir, LGB_SUMMARY)
    xgb_sum_path = os.path.join(args.xgb_dir, XGB_SUMMARY)

    df_lgb_raw = _safe_read_csv(lgb_sum_path)
    df_xgb_raw = _safe_read_csv(xgb_sum_path)

    if df_lgb_raw is None and df_xgb_raw is None:
        print("[ERROR] Neither summary exists, so analysis is not possible.")
        return

    df_lgb = _standardize_summary(df_lgb_raw, "lgb") if df_lgb_raw is not None else pd.DataFrame(columns=["ProgramID_encoded"])
    df_xgb = _standardize_summary(df_xgb_raw, "xgb") if df_xgb_raw is not None else pd.DataFrame(columns=["ProgramID_encoded"])

    merged = _merge_two_summaries(df_lgb, df_xgb)

    # 2) 用 bucket_metrics 补齐/增强（加权 Accuracy / ±1、桶稳定性）
    add_cols = [
        "Accuracy_lgb_weighted", "Accuracy+-1_lgb_weighted",
        "Accuracy_xgb_weighted", "Accuracy+-1_xgb_weighted",
        "worst_bucket_acc_lgb", "worst_bucket_acc_xgb",
        "bucket_count_lgb", "bucket_count_xgb",
        "buckets_below_thresh_lgb", "buckets_below_thresh_xgb",
    ]
    for c in add_cols:
        merged[c] = np.nan

    for i, row in merged.iterrows():
        pid = row["ProgramID_encoded"]
        if pd.isna(pid) or str(pid).strip().lower() == "others":
            continue

        # LGB
        if os.path.exists(os.path.join(args.lgb_dir, f"pid_{pid}_bucket_metrics.csv")):
            g = _collect_bucket_agg(args.lgb_dir, pid)
            merged.at[i, "Accuracy_lgb_weighted"]   = g["weighted_acc"]
            merged.at[i, "Accuracy+-1_lgb_weighted"] = g["weighted_acc_relax"]
            merged.at[i, "worst_bucket_acc_lgb"]    = g["worst_bucket_acc"]
            merged.at[i, "bucket_count_lgb"]        = g["bucket_count"]
            merged.at[i, "buckets_below_thresh_lgb"]= g["buckets_below_thresh"]

            # 若 summary 缺失 Accuracy，则用加权值补
            if np.isnan(merged.at[i, "Accuracy_lgb"]):
                merged.at[i, "Accuracy_lgb"] = g["weighted_acc"]
            if "Accuracy+-1_lgb" in merged.columns and np.isnan(merged.at[i, "Accuracy+-1_lgb"]):
                merged.at[i, "Accuracy+-1_lgb"] = g["weighted_acc_relax"]

        # XGB
        if os.path.exists(os.path.join(args.xgb_dir, f"pid_{pid}_bucket_metrics.csv")):
            g = _collect_bucket_agg(args.xgb_dir, pid)
            merged.at[i, "Accuracy_xgb_weighted"]   = g["weighted_acc"]
            merged.at[i, "Accuracy+-1_xgb_weighted"] = g["weighted_acc_relax"]
            merged.at[i, "worst_bucket_acc_xgb"]    = g["worst_bucket_acc"]
            merged.at[i, "bucket_count_xgb"]        = g["bucket_count"]
            merged.at[i, "buckets_below_thresh_xgb"]= g["buckets_below_thresh"]

            if np.isnan(merged.at[i, "Accuracy_xgb"]):
                merged.at[i, "Accuracy_xgb"] = g["weighted_acc"]
            if "Accuracy+-1_xgb" in merged.columns and np.isnan(merged.at[i, "Accuracy+-1_xgb"]):
                merged.at[i, "Accuracy+-1_xgb"] = g["weighted_acc_relax"]

    # 3) 决策
    cfg = dict(
        min_total_samples=args.min_total_samples,
        min_val_samples=args.min_val_samples,
        acc_strict_keep=args.acc_strict_keep,
        acc_relax_keep=args.acc_relax_keep,
        bucket_min_acc=args.bucket_min_acc,
        bucket_min_count=args.bucket_min_count,
        many_buckets=args.many_buckets,
    )

    decisions = []
    for i, row in merged.iterrows():
        pid = row["ProgramID_encoded"]
        if pd.isna(pid) or str(pid).strip().lower() == "others":
            decisions.append(("Others", "skip"))
            continue
        d, r = _decide_row(row, cfg)
        decisions.append((d, r))

    merged["Decision"] = [d for d, _ in decisions]
    merged["Reason"]   = [r for _, r in decisions]

    # 4) 导出
    out_dir = args.lgb_dir  # 集中放在 LGB 目录下
    os.makedirs(out_dir, exist_ok=True)
    merged_path = os.path.join(out_dir, "programid_selection_merged.csv")
    merged.to_csv(merged_path, index=False)
    print(f"[OK] Summary analysis written: {merged_path}")

    # 决策视图（精简列）
    keep_cols = [
        "ProgramID_encoded",
        "TrainSize_lgb", "ValSize_lgb", "Accuracy_lgb", "Accuracy+-1_lgb",
        "TrainSize_xgb", "ValSize_xgb", "Accuracy_xgb", "Accuracy+-1_xgb",
        "Accuracy_lgb_weighted", "Accuracy+-1_lgb_weighted",
        "Accuracy_xgb_weighted", "Accuracy+-1_xgb_weighted",
        "worst_bucket_acc_lgb", "worst_bucket_acc_xgb",
        "bucket_count_lgb", "bucket_count_xgb",
        "buckets_below_thresh_lgb", "buckets_below_thresh_xgb",
        "Decision", "Reason"
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    decisions_df = merged[keep_cols].copy()
    dec_path = os.path.join(out_dir, "programid_selection_decisions.csv")
    decisions_df.to_csv(dec_path, index=False)
    print(f"[OK] Decision recommendations have been written into: {dec_path}")

    # 三个清单
    keep_list   = merged.loc[merged["Decision"] == "Keep", "ProgramID_encoded"].dropna().tolist()
    watch_list  = merged.loc[merged["Decision"] == "Watchlist", "ProgramID_encoded"].dropna().tolist()
    merge_list  = merged.loc[merged["Decision"] == "MergeToOthers", "ProgramID_encoded"].dropna().tolist()

    with open(os.path.join(out_dir, "programid_selection_keep_list.json"), "w") as f:
        json.dump(keep_list, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "programid_selection_watchlist.json"), "w") as f:
        json.dump(watch_list, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "programid_selection_merge_list.json"), "w") as f:
        json.dump(merge_list, f, ensure_ascii=False, indent=2)

    print(f"[OK] The retain/observe/merge list has been exported to:{out_dir}")
    print(f"  Keep number: {len(keep_list)}")
    print(f"  Watchlist numner: {len(watch_list)}")
    print(f"  MergeToOthers number: {len(merge_list)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProgramID Trade-off recommendation script (based on v5.1 training output)")
    parser.add_argument("--lgb_dir", type=str, default=DEFAULT_LGB_DIR,
                        help="LightGBM Report directory (including summary and bucket_metrics)")
    parser.add_argument("--xgb_dir", type=str, default=DEFAULT_XGB_DIR,
                        help="XGBoost Report directory (including summary and bucket_metrics)")

    # 阈值
    parser.add_argument("--min_total_samples", type=float, default=DEFAULT_MIN_TOTAL_SAMPLES,
                        help="Train+Val total sample size lower limit; if lower than this value, merge Others")
    parser.add_argument("--min_val_samples", type=float, default=DEFAULT_MIN_VAL_SAMPLES,
                        help="ValSize lower limit; values below this are considered unstable")
    parser.add_argument("--acc_strict_keep", type=float, default=DEFAULT_ACC_STRICT_KEEP,
                        help="Retention threshold: Strictly Accuracy reaches this value (any model) is considered satisfactory")
    parser.add_argument("--acc_relax_keep", type=float, default=DEFAULT_ACC_RELAX_KEEP,
                        help="Retention threshold: Accuracy +-1 Reaching this value (for either model) means the target has been achieved")
    parser.add_argument("--bucket_min_acc", type=float, default=DEFAULT_BUCKET_MIN_ACC,
                        help="Minimum accuracy threshold within the bucket (for stability checks)")
    parser.add_argument("--bucket_min_count", type=int, default=DEFAULT_BUCKET_MIN_COUNT,
                        help="Minimum number of samples in a bucket participating in the minimum accuracy check")
    parser.add_argument("--many_buckets", type=int, default=DEFAULT_MANY_BUCKETS,
                        help="Determine the threshold for multiple buckets (used to identify suspicious PIDs with multiple buckets that are all substandard)")

    args = parser.parse_args()
    main(args)
