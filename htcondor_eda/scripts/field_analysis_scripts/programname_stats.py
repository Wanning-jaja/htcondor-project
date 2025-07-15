# -*- coding: utf-8 -*-

import pandas as pd
import os

# ==== 修改为你的实际输入文件路径 ====
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载数据 ===
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# === 提取 ProgramName 和 ProgramPath ===
#def extract_prog_name(cmd):
    #if pd.isnull(cmd) or cmd.strip() == '':
        #return ''
    #return os.path.basename(cmd.strip().split()[0])
    
def extract_prog_name(cmd):
    if pd.isnull(cmd) or cmd.strip() == '':
        return ''
    # 去掉末尾多余的斜杠，保证 basename 能正常提取
    path = cmd.strip().split()[0].rstrip('/')
    return os.path.basename(path)


def get_program_path(row):
    # 优先 Cmd，只有程序名/空则 fallback Submit_Cmd
    for col in ['Cmd', 'Submit_Cmd']:
        val = str(row.get(col, '')).strip()
        if '/' in val:
            return val.split()[0]
    # 兜底：如果 Cmd 是程序名
    if row.get('Cmd', '').strip():
        return row['Cmd'].strip().split()[0]
    return 'unknown'

df['ProgramName'] = df['Cmd'].apply(extract_prog_name)
df['ProgramPath'] = df.apply(get_program_path, axis=1)

# ==== 主表：ProgramName 汇总 ====
main_table = (
    df.groupby('ProgramName')
      .agg(
          cmd_count=('Cmd', 'count'),
          path_count=('ProgramPath', pd.Series.nunique),
          owners=('Owner', lambda x: ';'.join(sorted(set(x))))
      )
      .reset_index()
      .sort_values('cmd_count', ascending=False)
)
main_table.to_csv(os.path.join(OUTPUT_DIR, "ProgramName_summary.csv"), index=False)

# ==== 辅助表：ProgramName + ProgramPath 分布 ====
detail_table = (
    df.groupby(['ProgramName', 'ProgramPath'])
      .size()
      .reset_index(name='count')
      .sort_values(['ProgramName', 'count'], ascending=[True, False])
)
detail_table.to_csv(os.path.join(OUTPUT_DIR, "ProgramName_path_detail.csv"), index=False)

# ==== 日志输出 ====
print(f" finish")
print(f" ProgramName total: {df['ProgramName'].nunique()}")
print("  ProgramName empty:", (df['ProgramName'] == '').sum())
print(f" summary save to: {os.path.join(OUTPUT_DIR, 'ProgramName_summary.csv')}")
print(f" path detail save to: {os.path.join(OUTPUT_DIR, 'ProgramName_path_detail.csv')}")
