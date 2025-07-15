# -*- coding: utf-8 -*-

import pandas as pd
import os

# === �������·�� ===
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean_with_ownergroup.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean_with_ownergroup_filtered.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# === �������� ===
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
print(f" original total records : {len(df)}")

# === �ֶ�ת�������ڹ��ˣ�===
to_numeric_fields = ['RemoteWallClockTime', 'ExitCode', 'JobStatus', 'NumJobStarts']
for col in to_numeric_fields:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === Ӧ����ϴ���� ===
mask = (
    (df['RemoteWallClockTime'] > 0) &
    ((df['JobStatus'] != 3) | (df['ExitCode'] == 0)) &
    (df['NumJobStarts'] <= 1)
)
df_cleaned = df[mask]
print(f"total clean record : {len(df_cleaned)}")

# === ������� ===
df_cleaned.to_csv(OUTPUT_CSV, index=False)
print(f"data after clean save to : {OUTPUT_CSV}")
