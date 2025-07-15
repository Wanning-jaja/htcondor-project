# -*- coding: utf-8 -*-
import pandas as pd
import os

# === ·������ ===
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === �������� ===
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# === ת����ֵ�ֶ� ===
numeric_fields = [
    'RequestCpus', 'RequestMemory', 'RequestDisk',
    'ResidentSetSize_RAW', 'ImageSize_RAW',
    'RemoteWallClockTime', 'SubmitTime', 'JobCount'
]
for field in numeric_fields:
    df[field] = pd.to_numeric(df[field], errors='coerce')

# === ɾ�������ֶ� ===
df.drop(columns=['JobStatus', 'ExitCode', 'NumJobStarts', 'JobRunCount', 'GlobalJobId'], inplace=True, errors='ignore')

# === �����ϴ��Ľ�ģ�������� ===
output_path = os.path.join(OUTPUT_DIR, "model_features_v1.csv")
df.to_csv(output_path, index=False)
print("Features saved to:", output_path)
