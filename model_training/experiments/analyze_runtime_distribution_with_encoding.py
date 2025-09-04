# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import joblib

# === 路径配置 ===
input_path = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
encoder_path = "/home/master/wzheng/projects/model_training/data/encoders/ProgramID_encoder.joblib"
output_dir = "/home/master/wzheng/projects/model_training/reports/programID"
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "runtime_distribution_stats_with_encoded.csv")

# === 加载数据与编码器 ===
# 先把关键列当作字符串读取，避免因非数值字符（如 "undefined"）导致失败
dtype_str_cols = [
    "ResidentSetSize_RAW", "ImageSize_RAW",
    "RequestCpus", "RequestMemory", "RequestDisk",
    "RemoteWallClockTime", "ExitCode"
]
df = pd.read_csv(input_path, dtype={col: str for col in dtype_str_cols}, low_memory=False)

# 转换为数值类型（非法值如 'undefined' 会被转换为 NaN）
for col in dtype_str_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# === 编码 ProgramID ===
encoder = joblib.load(encoder_path)
df["ProgramID_encoded"] = df["ProgramID"].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# === 分组统计运行时间分布 ===
stats = df.groupby(["ProgramID", "ProgramID_encoded"])["RemoteWallClockTime"].agg([
    ("Count", "count"),
    ("Mean", "mean"),
    ("Std", "std"),
    ("Min", "min"),
    ("25%", lambda x: np.percentile(x.dropna(), 25)),
    ("Median", "median"),
    ("75%", lambda x: np.percentile(x.dropna(), 75)),
    ("Max", "max"),
    ("Skewness", lambda x: x.skew()),
    ("Kurtosis", lambda x: x.kurtosis())
]).reset_index()

# === 保存输出 ===
stats.to_csv(output_csv, index=False)
print(f"save to : {output_csv}")
