# -*- coding: utf-8 -*-  #

import pandas as pd
import os

# === 路径配置 ===
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/programID_grouped_cleaned.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/groupwise_runtime_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载数据 ===
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)

# === 类型转换 ===
for col in [
    'RemoteWallClockTime', 'CumulativeRemoteUserCpu', 'CumulativeRemoteSysCpu',
    'RequestCpus', 'RequestMemory', 'ResidentSetSize_RAW', 'ImageSize_RAW'
]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === 组合分析配置 ===
combinations = {
    "programID_owner": ["ProgramID", "Owner", "OwnerGroup"],
    "programID_resource": ["ProgramID", "RequestCpus", "RequestMemory"],
    "programID_memory": ["ProgramID", "ImageSize_RAW", "ResidentSetSize_RAW"],
    "programID_cpu_usage": ["ProgramID", "CumulativeRemoteUserCpu", "CumulativeRemoteSysCpu"]
}

summary_stats = {}
for name, cols in combinations.items():
    group = (
        df.groupby(cols)["RemoteWallClockTime"]
          .agg(["count", "mean", "median", "std", "min", "max"])
          .reset_index()
    )
    group.to_csv(os.path.join(OUTPUT_DIR, f"{name}_runtime_stats.csv"), index=False)
    summary_stats[name] = group

# === 相关性分析输出 ===
correlation_records = []
for name, group in summary_stats.items():
    temp = group.copy()
    numeric_cols = temp.select_dtypes(include='number').drop(columns=['count'], errors='ignore')
    corr_row = {'group': name}
    for col in numeric_cols.columns:
        if col != 'mean' and 'mean' in numeric_cols.columns:
            corr = temp[col].corr(temp['mean'])  # 与 mean 时间做相关性
            corr_row[col] = corr
    correlation_records.append(corr_row)

correlation_df = pd.DataFrame(correlation_records)
correlation_df.to_csv(os.path.join(OUTPUT_DIR, "runtime_correlation_summary.csv"), index=False)
print("save to", OUTPUT_DIR)
