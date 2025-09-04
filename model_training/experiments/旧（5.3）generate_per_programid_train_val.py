# -*- coding: utf-8 -*-

# 时间感知的分层切分（方案 C · 加强版）
# ==============================================
import os
import json
import numpy as np
import pandas as pd

# ===================== 路径与字段 =====================
FEATURES_CSV = "/home/master/wzheng/projects/model_training/data/model_features_v4.csv"
TOPN_JSON    = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"
OUTPUT_DIR   = "/home/master/wzheng/projects/model_training/data/top40_splits"
ALL_TRAIN    = "/home/master/wzheng/projects/model_training/data/40train.csv"
ALL_VAL      = "/home/master/wzheng/projects/model_training/data/40val.csv"

COL_PID   = "ProgramID_encoded"
COL_LABEL = "BucketLabel"     # 运行时长分桶标签（整数 0~9）
COL_TIME  = "SubmitTime"

# ===================== 超参 =====================
TEST_SIZE        = 0.20   # 验证集比例（桶内按尾部取比例）
MIN_PER_SPLIT    = 1      # 每个桶在 train/val 至少样本数
SMALL_POLICY     = "all_to_train"  # {"all_to_train","merge_to_others"}
TIME_GAP         = 0      # 可选时间净空带（与 SubmitTime 同单位），0 表示关闭
RANDOM_STATE     = 42
np_random = np.random.RandomState(RANDOM_STATE)

# ===================== 工具函数 =====================
def _time_tail_split_bucket(df_bucket: pd.DataFrame,
                            test_size: float,
                            min_per_split: int,
                            time_col: str,
                            time_gap: int = 0):
#    """在单一 ProgramID 的单一 Bucket 内执行“按时间尾部切分”。返回 (train_df, val_df)；若样本不足以满足 min 约束，返回 None。"""
    if df_bucket.empty:
        return None
    df_bucket = df_bucket.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_bucket)

    # 基线：按比例取尾部
    val_cnt = max(min_per_split, int(round(n * test_size)))
    train_cnt = n - val_cnt

    # 同时保证 train/val 均 >= min
    if train_cnt < min_per_split:
        return None

    train_df = df_bucket.iloc[:train_cnt].copy()
    val_df   = df_bucket.iloc[train_cnt:].copy()

    # 可选：加入时间净空带（soft 约束）
    if time_gap and not val_df.empty and not train_df.empty:
        min_val_time = val_df[time_col].min()
        gap_border   = min_val_time - time_gap
        move_mask = train_df[time_col] >= gap_border
        if move_mask.any():
            moved  = train_df.loc[move_mask]
            remain = train_df.loc[~move_mask]
            if len(remain) >= min_per_split:  # 不破坏最小样本约束时才移动
                train_df = remain
                val_df   = pd.concat([moved, val_df], ignore_index=True)
    return train_df, val_df


def split_one_pid(df_pid: pd.DataFrame,
                  label_col: str,
                  test_size: float,
                  min_per_split: int,
                  time_col: str,
                  time_gap: int = 0):
#    """对单个 ProgramID，逐桶按时间尾部切分；聚合所有桶的 train/val。任一桶无法满足 min 约束则返回 None。"""
    trains, vals = [], []
    for _, g in df_pid.groupby(label_col):
        res = _time_tail_split_bucket(g, test_size, min_per_split, time_col, time_gap)
        if res is None:
            return None
        tr, va = res
        trains.append(tr); vals.append(va)
    return (pd.concat(trains, ignore_index=True) if trains else df_pid.iloc[0:0].copy(),
            pd.concat(vals,   ignore_index=True) if vals   else df_pid.iloc[0:0].copy())


# ===================== 主流程 =====================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读入数据与 TopN 列表
    df = pd.read_csv(FEATURES_CSV)
    with open(TOPN_JSON, "r") as f:
        top_programids = set(json.load(f))

    # ==== 若没有 BucketLabel，就按 RemoteWallClockTime（秒）生成 ====
    if COL_LABEL not in df.columns:
        if "RemoteWallClockTime" not in df.columns:
            raise ValueError("Unable to generate BucketLabel: RemoteWallClockTime column is missing.")
        df["RemoteWallClockTime"] = pd.to_numeric(df["RemoteWallClockTime"], errors="coerce")

        def get_bucket_label(sec):
            if pd.isna(sec): return np.nan
            sec = float(sec)
            if sec < 600: return 0
            elif sec < 1800: return 1
            elif sec < 3600: return 2
            elif sec < 10800: return 3
            elif sec < 21600: return 4
            elif sec < 43200: return 5

            else: return 6

        df[COL_LABEL] = df["RemoteWallClockTime"].apply(get_bucket_label).astype("Int64")

    # ==== 基本校验 & 类型规范化 ====
    # 时间列转数值，去掉 SubmitTime 或 BucketLabel 缺失的行（无法分桶或排序的样本）
    df[COL_TIME] = pd.to_numeric(df[COL_TIME], errors="coerce").astype("Int64")
    for col in [COL_PID, COL_LABEL, COL_TIME]:
        if col not in df.columns:
            raise ValueError(f"Missing required columns: {col}")
    df = df.dropna(subset=[COL_TIME, COL_LABEL]).copy()

    # 为可读性排序（不影响尾部分割逻辑）
    df = df.sort_values(by=COL_TIME).reset_index(drop=True)

    # TopN / Others 划分
    is_top   = df[COL_PID].isin(top_programids)
    df_top    = df.loc[is_top].copy()
    df_others = df.loc[~is_top].copy()

    train_all, val_all = [], []
    small_merge_pool = []   # SMALL_POLICY == "merge_to_others" 时使用

    # ===== TopN：逐 PID 执行“时间尾部分层” =====
    for pid in sorted(top_programids):
        df_pid = df_top.loc[df_top[COL_PID] == pid].copy()
        tr_path = os.path.join(OUTPUT_DIR, f"train_top{pid}.csv")
        va_path = os.path.join(OUTPUT_DIR, f"val_top{pid}.csv")

        if df_pid.empty:
            print(f"[WARN] PID {pid}: No sample, skip.")
            pd.DataFrame(columns=df.columns).to_csv(tr_path, index=False)
            pd.DataFrame(columns=df.columns).to_csv(va_path, index=False)
            continue

        res = split_one_pid(df_pid, COL_LABEL, TEST_SIZE, MIN_PER_SPLIT, COL_TIME, TIME_GAP)

        if res is None:
            if SMALL_POLICY == "merge_to_others":
                small_merge_pool.append(df_pid)
                pd.DataFrame(columns=df.columns).to_csv(tr_path, index=False)
                pd.DataFrame(columns=df.columns).to_csv(va_path, index=False)
                print(f"[SMALL->OTHERS] PID {pid}: Insufficient samples in some bucket; merged into Others.")
            else:
                df_empty = df_pid.iloc[0:0].copy()
                df_pid.to_csv(tr_path, index=False)
                df_empty.to_csv(va_path, index=False)
                train_all.append(df_pid)
                print(f"[SMALL->TRAIN] PID {pid}: Insufficient per-bucket samples; all go to train, val empty.")
            continue

        tr_df, va_df = res
        tr_df.to_csv(tr_path, index=False)
        va_df.to_csv(va_path, index=False)
        train_all.append(tr_df)
        val_all.append(va_df)

        tr_stats = tr_df[COL_LABEL].value_counts().to_dict()
        va_stats = va_df[COL_LABEL].value_counts().to_dict()
        latest_tr = tr_df[COL_TIME].max() if not tr_df.empty else None
        earliest_va = va_df[COL_TIME].min() if not va_df.empty else None
        print(f"PID {pid}: Train={len(tr_df)} {tr_stats} | Val={len(va_df)} {va_stats} | time_cut=({latest_tr} -> {earliest_va})")

    # ===== Others（含合并的小样本 PID） =====
    if SMALL_POLICY == "merge_to_others" and small_merge_pool:
        df_others = pd.concat([df_others] + small_merge_pool, ignore_index=True)

    if not df_others.empty:
        trains, vals = [], []
        for (_, _), gpid in df_others.groupby([COL_PID, COL_LABEL]):
            res = _time_tail_split_bucket(gpid, TEST_SIZE, MIN_PER_SPLIT, COL_TIME, TIME_GAP)
            if res is None:
                trains.append(gpid)  # 不足则全进 train
                continue
            tr, va = res
            trains.append(tr); vals.append(va)

        train_others = pd.concat(trains, ignore_index=True) if trains else df_others.iloc[0:0].copy()
        val_others   = pd.concat(vals,   ignore_index=True) if vals   else df_others.iloc[0:0].copy()

        train_others.to_csv(os.path.join(OUTPUT_DIR, "train_topOthers.csv"), index=False)
        val_others.to_csv(os.path.join(OUTPUT_DIR, "val_topOthers.csv"), index=False)
        train_all.append(train_others)
        val_all.append(val_others)
        print(f"Others: Train={len(train_others)} | Val={len(val_others)}")

    # ===== 汇总输出（全体） =====
    train_all_df = pd.concat(train_all, ignore_index=True) if train_all else df.iloc[0:0].copy()
    val_all_df   = pd.concat(val_all,   ignore_index=True) if val_all   else df.iloc[0:0].copy()

    # 再次按时间排序（不改变切分结果）
    if not train_all_df.empty:
        train_all_df = train_all_df.sort_values(by=COL_TIME).reset_index(drop=True)
    if not val_all_df.empty:
        val_all_df = val_all_df.sort_values(by=COL_TIME).reset_index(drop=True)

    os.makedirs(os.path.dirname(ALL_TRAIN), exist_ok=True)
    train_all_df.to_csv(ALL_TRAIN, index=False)
    val_all_df.to_csv(ALL_VAL, index=False)

    # 打印总体统计
    print(f" Overall: Train={len(train_all_df)} | Val={len(val_all_df)}")
    if not train_all_df.empty:
        print("Train bucket counts:", train_all_df[COL_LABEL].value_counts().to_dict())
    if not val_all_df.empty:
        print("Val bucket counts:", val_all_df[COL_LABEL].value_counts().to_dict())
    print("output path :", OUTPUT_DIR)
