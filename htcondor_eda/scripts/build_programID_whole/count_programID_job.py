# -*- coding: utf-8 -*-
import pandas as pd

# === 输入路径 ===
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramID_wholedata_group/programID_grouped_cleaned.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramID_wholedata_group/programID_job_counts.csv"

# === 加载数据 ===
df = pd.read_csv(INPUT_CSV)

# === 按 ProgramID 统计任务数 ===
programid_counts = df['ProgramID'].value_counts().reset_index()
programid_counts.columns = ['ProgramID', 'JobCount']
programid_counts = programid_counts.sort_values(by='JobCount', ascending=False)

# === 输出结果 ===
programid_counts.to_csv(OUTPUT_CSV, index=False)
print("every ProgramID jobcount save to: ", OUTPUT_CSV)
