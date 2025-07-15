# -*- coding: utf-8 -*-

import pandas as pd

# ����·��
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/ProgramName_path_detail.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/ProgramName_unique_distribution.csv"

# ��ȡ����
df = pd.read_csv(INPUT_CSV)

# �� ProgramName �ۺ���ҵ��
program_counts = df.groupby("ProgramName")["count"].sum().sort_values(ascending=False)
program_counts = program_counts.reset_index()

# ���ͳ����Ϣ
print(f"unique ProgramName total: {program_counts.shape[0]}")
print(f"Top 200 programname count:\n{program_counts.head(200)}")
print(program_counts["count"].describe())

# ����Ϊ csv ����������/��ͼ
program_counts.to_csv(OUTPUT_CSV, index=False)
print(f"save unique ProgramName distribution to: {OUTPUT_CSV}")
