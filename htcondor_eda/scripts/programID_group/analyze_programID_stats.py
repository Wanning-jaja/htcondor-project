# -*- coding: utf-8 -*-  #

import pandas as pd

# 输入路径（你前面成功生成的任务级数据）
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/programID_grouped_cleaned.csv"

# 输出路径（ProgramID 级别统计结果）
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/programID_stats_summary.csv"

# === 更稳健的读取方式 ===
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)

# === 显式转换 RemoteWallClockTime 为数值型 ===
df["RemoteWallClockTime"] = pd.to_numeric(df["RemoteWallClockTime"], errors="coerce")

# === 按 ProgramID 聚合运行时间统计指标 ===
stats = (
    df.groupby("ProgramID")["RemoteWallClockTime"]
      .agg(["count", "mean", "median", "std", "min", "max"])
      .reset_index()
      .sort_values(by="count", ascending=False)
)

# === 输出为 CSV 文件 ===
stats.to_csv(OUTPUT_CSV, index=False)
print("programID_stats_summary saved to:", OUTPUT_CSV)