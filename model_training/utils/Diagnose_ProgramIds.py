# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# === ���ò��� ===
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
ENCODER_PATH = "/home/master/wzheng/projects/model_training/data/encoders/ProgramID_encoder.joblib"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/reports/diagnostics/"
PROGRAM_IDS_TO_ANALYZE = [126, 103, 10, 176, 134, 168, 200, 99, 110]

# === �������·�� ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === ��������������� ===
df = pd.read_csv(INPUT_CSV, low_memory=False)
df.columns = df.columns.str.strip()
encoder = joblib.load(ENCODER_PATH)

# ��� ProgramID �в����ڣ�ֱ�ӱ���
if "ProgramID" not in df.columns:
    raise ValueError("ProgramID column not found. Please confirm whether the original data contains the original ProgramID.")

# ���� ProgramID => ProgramID_encoded
# ���ڲ��� encoder.classes_ �е�ֵ�����Ϊ -1
encode_map = {label: int(encoder.transform([label])[0]) for label in encoder.classes_}
df["ProgramID_encoded"] = df["ProgramID"].map(encode_map).fillna(-1).astype(int)

# === ����ÿ�� ProgramID ===
df = df[df["ProgramID_encoded"].isin(PROGRAM_IDS_TO_ANALYZE)].copy()
df["RemoteWallClockTime"] = pd.to_numeric(df["RemoteWallClockTime"], errors="coerce")

for pid in PROGRAM_IDS_TO_ANALYZE:
    df_pid = df[df["ProgramID_encoded"] == pid]
    subdir = os.path.join(OUTPUT_DIR, f"programID_{pid}")
    os.makedirs(subdir, exist_ok=True)

    # === ����ͳ����Ϣ ===
    stats = {
        "Count": len(df_pid),
        "Mean": df_pid["RemoteWallClockTime"].mean(),
        "Std": df_pid["RemoteWallClockTime"].std(),
        "Min": df_pid["RemoteWallClockTime"].min(),
        "25%": np.percentile(df_pid["RemoteWallClockTime"], 25),
        "Median": df_pid["RemoteWallClockTime"].median(),
        "75%": np.percentile(df_pid["RemoteWallClockTime"], 75),
        "Max": df_pid["RemoteWallClockTime"].max(),
        "Skewness": df_pid["RemoteWallClockTime"].skew(),
        "Kurtosis": df_pid["RemoteWallClockTime"].kurtosis()
    }
    pd.DataFrame([stats]).to_csv(os.path.join(subdir, "runtime_stats.csv"), index=False)

    # === ��ͼ���� ===
    plt.figure(figsize=(10, 6))
    sns.histplot(df_pid["RemoteWallClockTime"], bins=100, kde=True)
    plt.title(f"ProgramID {pid} - Runtime Distribution")
    plt.xlabel("RemoteWallClockTime (s)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "hist_linear.png"))
    plt.close()

    # === ���� log �ֲ�ͼ ===
    df_log = df_pid[df_pid["RemoteWallClockTime"] > 0].copy()
    df_log["log_time"] = np.log10(df_log["RemoteWallClockTime"])

    plt.figure(figsize=(10, 6))
    sns.histplot(df_log["log_time"], bins=100, kde=True)
    plt.title(f"ProgramID {pid} - Log10 Runtime Distribution")
    plt.xlabel("log10(RemoteWallClockTime)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(subdir, "hist_log10.png"))
    plt.close()

    # === ��� Top Owner��Path������ �ֲ��������ֶΣ� ===
    top_fields = ["Owner", "ProgramID", "Arguments"]
    for field in top_fields:
        if field in df_pid.columns:
            value_counts = df_pid[field].value_counts().head(10)
            value_counts.to_csv(os.path.join(subdir, f"top_{field}.csv"))

print("save to :", OUTPUT_DIR)
