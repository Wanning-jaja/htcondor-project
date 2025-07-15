# -*- coding: utf-8 -*-

import pandas as pd

# ���ò���
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/ProgramName_path_detail.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/TopN_ProgramName_path_distribution.csv"
TOPN = 50    # ���ǰ N �ĳ�����
TOPM = 6    # ÿ�����������ǰ M ·���ֲ�

df = pd.read_csv(INPUT_CSV)

# ͳ�Ƴ���������������
prog_total = df.groupby('ProgramName')['count'].sum().sort_values(ascending=False)
topN_names = prog_total.head(TOPN).index.tolist()

rows = []
for pname in topN_names:
    sub = df[df['ProgramName'] == pname].sort_values('count', ascending=False).head(TOPM)
    for _, row in sub.iterrows():
        rows.append({
            "ProgramName": pname,
            "ProgramPath": row['ProgramPath'],
            "count": row['count'],
        })
    # ����ָ��У������˹�����
    rows.append({"ProgramName": "---", "ProgramPath": "---", "count": "---"})

# ������
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Top{TOPN}  Top{TOPM} to: {OUTPUT_CSV}")
