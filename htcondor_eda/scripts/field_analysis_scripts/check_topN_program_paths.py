# -*- coding: utf-8 -*-

import pandas as pd

# 配置参数
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/ProgramName_path_detail.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/TopN_ProgramName_path_distribution.csv"
TOPN = 50    # 输出前 N 的程序名
TOPM = 6    # 每个程序名输出前 M 路径分布

df = pd.read_csv(INPUT_CSV)

# 统计程序名的总任务量
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
    # 插入分割行，便于人工查阅
    rows.append({"ProgramName": "---", "ProgramPath": "---", "count": "---"})

# 保存结果
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"Top{TOPN}  Top{TOPM} to: {OUTPUT_CSV}")
