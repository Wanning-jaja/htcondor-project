# -*- coding: utf-8 -*-  #

import pandas as pd

# ����·������ǰ��ɹ����ɵ��������ݣ�
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/programID_grouped_cleaned.csv"

# ���·����ProgramID ����ͳ�ƽ����
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/programID_stats_summary.csv"

# === ���Ƚ��Ķ�ȡ��ʽ ===
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)

# === ��ʽת�� RemoteWallClockTime Ϊ��ֵ�� ===
df["RemoteWallClockTime"] = pd.to_numeric(df["RemoteWallClockTime"], errors="coerce")

# === �� ProgramID �ۺ�����ʱ��ͳ��ָ�� ===
stats = (
    df.groupby("ProgramID")["RemoteWallClockTime"]
      .agg(["count", "mean", "median", "std", "min", "max"])
      .reset_index()
      .sort_values(by="count", ascending=False)
)

# === ���Ϊ CSV �ļ� ===
stats.to_csv(OUTPUT_CSV, index=False)
print("programID_stats_summary saved to:", OUTPUT_CSV)