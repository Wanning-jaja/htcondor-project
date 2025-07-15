# -*- coding: utf-8 -*-
import pandas as pd

# === ����·�� ===
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramID_wholedata_group/programID_grouped_cleaned.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramID_wholedata_group/programID_job_counts.csv"

# === �������� ===
df = pd.read_csv(INPUT_CSV)

# === �� ProgramID ͳ�������� ===
programid_counts = df['ProgramID'].value_counts().reset_index()
programid_counts.columns = ['ProgramID', 'JobCount']
programid_counts = programid_counts.sort_values(by='JobCount', ascending=False)

# === ������ ===
programid_counts.to_csv(OUTPUT_CSV, index=False)
print("every ProgramID jobcount save to: ", OUTPUT_CSV)
