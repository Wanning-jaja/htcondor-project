# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# === 配置 ===
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/model_features_v1.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/data/"
ENCODER_DIR = os.path.join(OUTPUT_DIR, "encoders")
os.makedirs(ENCODER_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载原始数据 ===
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

# === Label Encoding: 程序标识与用户标识 ===
label_cols = ['Owner', 'OwnerGroup', 'ProgramID', 'ProgramName']
for col in label_cols:
    df[col] = df[col].fillna("missing")
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    joblib.dump(le, os.path.join(ENCODER_DIR, f"{col}_encoder.joblib"))

# === 处理 Cmd（来自 ProgramName） ===
df['Cmd'] = df['ProgramName'].fillna('unknown').astype(str)
le_cmd = LabelEncoder()
df['Cmd_encoded'] = le_cmd.fit_transform(df['Cmd'])
joblib.dump(le_cmd, os.path.join(ENCODER_DIR, "Cmd_encoder.joblib"))

# === SubmitTime 装换成周中日期/小时 ===
df['SubmitTime'] = pd.to_numeric(df['SubmitTime'], errors='coerce')
df['SubmitHour'] = pd.to_datetime(df['SubmitTime'], unit='s', errors='coerce').dt.hour.fillna(-1).astype(int)
df['SubmitWeekday'] = pd.to_datetime(df['SubmitTime'], unit='s', errors='coerce').dt.weekday.fillna(-1).astype(int)

# === 数值类特征 ===
numeric_cols = [
    'RequestCpus', 'RequestMemory', 'RequestDisk',
    'ResidentSetSize_RAW', 'ImageSize_RAW',
    'JobCount'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df[col + "_log1p"] = np.log1p(df[col])

# === 目标值处理 ===
df['RemoteWallClockTime'] = pd.to_numeric(df['RemoteWallClockTime'], errors='coerce')

# === 选择最终特征 ===
model_df = df[
    [col + "_encoded" for col in label_cols] +
    ['Cmd_encoded', 'SubmitHour', 'SubmitWeekday'] +
    [col + "_log1p" for col in numeric_cols] +
    ['SubmitTime', 'RemoteWallClockTime']
].copy()

# === 输出 ===
output_path = os.path.join(OUTPUT_DIR, "model_features_v3.csv")
model_df.to_csv(output_path, index=False)
print("\u2714\ufe0f Features v3 saved to:", output_path)
print("\u2714\ufe0f Encoders saved to:", ENCODER_DIR)
