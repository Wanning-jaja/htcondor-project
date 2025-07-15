# -*- coding: utf-8 -*-

import pandas as pd
import os
import re

# ===== 路径配置（请根据环境修改）=====
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean_filtered.csv"
MAPPING_CSV = "/home/master/wzheng/projects/htcondor_eda/scripts/ProgramCategory/program_name_mapping.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ProgramName_unified_grouped.csv")
LOWFREQ_ONLY_CSV = os.path.join(OUTPUT_DIR, "ProgramName_lowfreq_only.csv")
OTHERS_ONLY_CSV = os.path.join(OUTPUT_DIR, "ProgramName_others_only.csv")

# ===== 聚类参数 =====
LOW_COUNT_THRESHOLD = 300
MAX_NAMES_DISPLAY = 5

# ===== 加载主数据（清洗后）=====
df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
df['Cmd'] = df['Cmd'].astype(str).str.strip()

# ===== 提取 ProgramName =====
df['ProgramName'] = df['Cmd'].apply(lambda x: os.path.basename(x.split()[0]) if '/' in x else x.split()[0] if x else '')

# ===== 聚合频次 =====
program_counts = df['ProgramName'].value_counts().reset_index()
program_counts.columns = ['ProgramName', 'count']

# ===== 加载 mapping 表 =====
mapping_df = pd.read_csv(MAPPING_CSV)

def get_category(name):
    if pd.isnull(name) or str(name).strip() == "":
        return 'undefined'
    for _, row in mapping_df.iterrows():
        pattern = row['ProgramNamePattern']
        if re.match(pattern, str(name)):
            return row['ProgramCategory']
    return 'others'  # 明确归入“others”用于下一步分析

program_counts['ProgramCategory'] = program_counts['ProgramName'].apply(get_category)

# ===== 分组处理 =====
main_rows, lowfreq_rows, others_rows = [], [], []

for _, row in program_counts.iterrows():
    cat, name, cnt = row['ProgramCategory'], row['ProgramName'], row['count']
    if cat == 'others':
        others_rows.append({'ProgramCategory': cat, 'ProgramName': name, 'count': cnt})
    elif cnt <= LOW_COUNT_THRESHOLD:
        lowfreq_rows.append({'ProgramCategory': 'low_freq', 'ProgramName': name, 'count': cnt})
    else:
        main_rows.append({'ProgramCategory': cat, 'ProgramName': name, 'count': cnt})

main_df = pd.DataFrame(main_rows)
lowfreq_df = pd.DataFrame(lowfreq_rows)
others_df = pd.DataFrame(others_rows)

# ===== 汇总主表（ProgramCategory → ProgramNames） =====
def format_names(names, maxn=5):
    names = sorted(set(str(n) for n in names))
    result = names[:maxn]
    tail = f"... (total {len(names)})" if len(names) > maxn else ""
    return ";".join(result) + ((";" + tail) if tail else "")

def names_agg(df_):
    if df_.empty:
        return pd.DataFrame(columns=["ProgramCategory", "ProgramNames", "count"])
    return df_.groupby('ProgramCategory').agg(
        ProgramNames=('ProgramName', lambda x: format_names(x, MAX_NAMES_DISPLAY)),
        count=('count','sum')
    ).reset_index().sort_values('count', ascending=False)

grouped_main = names_agg(main_df)
grouped_lowfreq = names_agg(lowfreq_df)
grouped_others = names_agg(others_df)

# ===== 保存输出 =====
all_grouped = pd.concat([grouped_main, grouped_lowfreq, grouped_others], ignore_index=True)
all_grouped.to_csv(OUTPUT_CSV, index=False)
print(f"ProgramName save to: {OUTPUT_CSV}")

lowfreq_df[['ProgramName', 'count']].to_csv(LOWFREQ_ONLY_CSV, index=False)
others_df[['ProgramName', 'count']].to_csv(OTHERS_ONLY_CSV, index=False)
print(f"lowfrequency save to : {LOWFREQ_ONLY_CSV}")
print(f"others save to : {OTHERS_ONLY_CSV}")
