# -*- coding: utf-8 -*-

import pandas as pd

# 配置路径
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/ProgramName_path_detail.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/ProgramName_unique_distribution.csv"

# 读取数据
df = pd.read_csv(INPUT_CSV)

# 按 ProgramName 聚合作业量
program_counts = df.groupby("ProgramName")["count"].sum().sort_values(ascending=False)
program_counts = program_counts.reset_index()

# 输出统计信息
print(f"unique ProgramName total: {program_counts.shape[0]}")
print(f"Top 200 programname count:\n{program_counts.head(200)}")
print(program_counts["count"].describe())

# 保存为 csv 供后续分析/作图
program_counts.to_csv(OUTPUT_CSV, index=False)
print(f"save unique ProgramName distribution to: {OUTPUT_CSV}")
