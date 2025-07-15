# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

# === 输入输出路径 ===
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/programID_grouped_cleaned.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/combined_field_time_correlation.csv"

# === 加载数据 ===
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)

# === 所有涉及字段 ===
fields_to_convert = [
    'RequestCpus', 'RequestMemory', 'RequestDisk',
    'CumulativeRemoteUserCpu', 'CumulativeRemoteSysCpu',
    'ResidentSetSize_RAW', 'ImageSize_RAW',
    'JobRunCount', 'NumJobStarts',
    'ExitCode', 'JobStatus',
    'RemoteWallClockTime'
]

# === 转换为数值类型 ===
df_num = df[fields_to_convert].apply(pd.to_numeric, errors='coerce').dropna()

# === 设置目标字段 ===
target = df_num['RemoteWallClockTime']

# === 单字段相关性分析 ===
results = []
for col in df_num.columns:
    if col != 'RemoteWallClockTime':
        corr = df_num[col].corr(target)
        results.append({
            "GroupName": "Single",
            "Factor1": col,
            "Factor2": "(single)",
            "CorrelationWithTime": round(corr, 4)
        })

# === 组合字段配置 ===
combinations = {
    "programID_owner": ["Owner", "OwnerGroup"],
    "programID_resource": ["RequestCpus", "RequestMemory"],
    "programID_memory": ["ImageSize_RAW", "ResidentSetSize_RAW"],
    "programID_cpu_usage": ["CumulativeRemoteUserCpu", "CumulativeRemoteSysCpu"]
}

# === 组合字段相关性（用两个字段的平均值表示组合）===
for group_name, cols in combinations.items():
    if all(c in df_num.columns for c in cols):
        composite = df_num[cols].mean(axis=1)
        corr = composite.corr(target)
        results.append({
            "GroupName": group_name,
            "Factor1": cols[0],
            "Factor2": cols[1] if len(cols) > 1 else "(no)",
            "CorrelationWithTime": round(corr, 4)
        })

# === 结果保存 ===
corr_df = pd.DataFrame(results)
corr_df = corr_df.sort_values(by="CorrelationWithTime", ascending=False)
corr_df.to_csv(OUTPUT_CSV, index=False)

print("save to ", OUTPUT_CSV)
