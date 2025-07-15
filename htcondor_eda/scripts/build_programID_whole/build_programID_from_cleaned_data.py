# -*- coding: utf-8 -*-
import pandas as pd
import os
import re

# === 路径配置 ===
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean_with_ownergroup_filtered.csv"
MAPPING_CSV = "/home/master/wzheng/projects/htcondor_eda/scripts/build_programID_whole/program_name_mapping.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/ProgramID_wholedata_group/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载数据 ===
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
mapping_df = pd.read_csv(MAPPING_CSV)

# === 提取 ProgramName ===
df['ProgramName'] = df['Cmd'].apply(lambda x: os.path.basename(str(x).strip().split()[0]) if str(x).strip() else '')

# === 提取 ProgramPath（前4级） ===
def get_program_path(row):
    for col in ['Cmd', 'SUBMIT_Cmd']:
        val = str(row.get(col, '')).strip()
        if '/' in val:
            return val.split()[0]
    return row.get('Cmd', '').strip().split()[0] if row.get('Cmd', '').strip() else 'unknown'

df['ProgramPath'] = df.apply(get_program_path, axis=1)
df['ProgramPath4'] = df['ProgramPath'].apply(lambda x: '/'.join(x.strip().split('/')[:4]) if x else 'unknown')

# === 正则匹配 ProgramName 分类 ===
regex_pairs = list(zip(mapping_df["ProgramNamePattern"], mapping_df["GroupName"]))

def match_category(name, regex_pairs):
    for pattern, group in regex_pairs:
        try:
            if re.fullmatch(pattern, name):
                return group
        except:
            continue
    return "Uncategorized"

df['ProgramNameCategory'] = df['ProgramName'].apply(lambda x: match_category(x, regex_pairs))

# === 构建 ProgramID ===
df['ProgramID'] = df['ProgramNameCategory'] + "::" + df['ProgramPath4']

# 新增：提取 SubmitTime（任务提交时间）
df['SubmitTime'] = df['GlobalJobId'].str.split('#').str[-1].astype(int)

# === 保留字段 ===
fields_keep = [
    'ProgramID', 'ProgramName', 'ProgramPath4', 'Owner', 'OwnerGroup',
    'RequestCpus', 'RequestMemory', 'RequestDisk',
    'ResidentSetSize_RAW', 'ImageSize_RAW', 'NumJobStarts', 'JobRunCount',
    'JobStatus', 'ExitCode', 'GlobalJobId','RemoteWallClockTime','SubmitTime' 
]
df_out = df[fields_keep].copy()

# === 按 ProgramID 聚合任务数，排序 ===
program_counts = df_out['ProgramID'].value_counts().reset_index()
program_counts.columns = ['ProgramID', 'JobCount']
df_out = df_out.merge(program_counts, on='ProgramID')
df_out = df_out.sort_values(by=['JobCount', 'ProgramID'], ascending=[False, True])

# === 输出为 CSV ===
output_path = os.path.join(OUTPUT_DIR, "programID_grouped_cleaned(withOwnerGroup).csv")
df_out.to_csv(output_path, index=False)
print("ProgramID finished ", output_path)
