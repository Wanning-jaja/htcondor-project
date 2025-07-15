# -*- coding: utf-8 -*-  #

import pandas as pd
import os

# === 路径配置（你需要根据本地环境替换） ===
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/programID_Group/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载数据 ===
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# === 提取 ProgramName ===
df['ProgramName'] = df['Cmd'].apply(lambda x: os.path.basename(str(x).strip().split()[0]) if str(x).strip() else '')

# === 提取 ProgramPath（前4级）===
def get_program_path(row):
    for col in ['Cmd', 'SUBMIT_Cmd']:
        val = str(row.get(col, '')).strip()
        if '/' in val:
            return val.split()[0]
    return row.get('Cmd', '').strip().split()[0] if row.get('Cmd', '').strip() else 'unknown'
df['ProgramPath'] = df.apply(get_program_path, axis=1)
df['ProgramPath4'] = df['ProgramPath'].apply(lambda x: '/'.join(x.strip().split('/')[:4]) if x else 'unknown')

# === 构建 ProgramID ===
df['ProgramID'] = df['ProgramName'] + "::" + df['ProgramPath4']

# === 转换字段类型（用于筛选）===
df['RemoteWallClockTime'] = pd.to_numeric(df['RemoteWallClockTime'], errors='coerce')
df['ExitCode'] = pd.to_numeric(df['ExitCode'], errors='coerce')
df['JobStatus'] = pd.to_numeric(df['JobStatus'], errors='coerce')
df['NumJobStarts'] = pd.to_numeric(df['NumJobStarts'], errors='coerce')

# === 数据清洗规则 ===
mask = (df['RemoteWallClockTime'] > 0) & \
       ((df['JobStatus'] != 3) | (df['ExitCode'] == 0)) & \
       (df['NumJobStarts'] <= 1)
df = df[mask]

# === 字段保留（仅你指定的字段）===
fields_keep = [
    'ProgramID', 'ProgramName', 'ProgramPath4', 'Owner', 'OwnerGroup',
    'RequestCpus', 'RequestMemory', 'RequestDisk',
    'CumulativeRemoteUserCpu', 'CumulativeRemoteSysCpu', 'CumulativeSuspensionTime',
    'ResidentSetSize_RAW', 'ImageSize_RAW', 'NumJobStarts', 'JobRunCount',
    'JobStatus', 'ExitCode', 'RemoteWallClockTime'
]
df_out = df[fields_keep].copy()

# === 按 ProgramID 聚合任务数，排序 ===
program_counts = df_out['ProgramID'].value_counts().reset_index()
program_counts.columns = ['ProgramID', 'JobCount']
df_out = df_out.merge(program_counts, on='ProgramID')
df_out = df_out.sort_values(by=['JobCount', 'ProgramID'], ascending=[False, True])

# === 输出为 CSV ===
output_path = os.path.join(OUTPUT_DIR, "programID_grouped_cleaned.csv")
df_out.to_csv(output_path, index=False)
print("output finished", output_path)
