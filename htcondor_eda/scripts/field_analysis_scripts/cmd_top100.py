# -*- coding: utf-8 -*-

import os
import pandas as pd

INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# CMD ×Ö¶Î Top-100 Í³¼Æ
top100_cmd = df['Cmd'].value_counts().head(100).reset_index()
top100_cmd.columns = ['Cmd', 'count']

# ±£´æ

top100_cmd.to_csv(os.path.join(OUTPUT_DIR, "top100_cmd.csv"), index=False)

print("? top100_cmd analysis completed.")