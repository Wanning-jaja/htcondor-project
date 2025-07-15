# -*- coding: utf-8 -*-

import pandas as pd
import os

# ==== �޸�Ϊ���ʵ�������ļ�·�� ====
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === �������� ===
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# === ��ȡ ProgramName �� ProgramPath ===
#def extract_prog_name(cmd):
    #if pd.isnull(cmd) or cmd.strip() == '':
        #return ''
    #return os.path.basename(cmd.strip().split()[0])
    
def extract_prog_name(cmd):
    if pd.isnull(cmd) or cmd.strip() == '':
        return ''
    # ȥ��ĩβ�����б�ܣ���֤ basename ��������ȡ
    path = cmd.strip().split()[0].rstrip('/')
    return os.path.basename(path)


def get_program_path(row):
    # ���� Cmd��ֻ�г�����/���� fallback Submit_Cmd
    for col in ['Cmd', 'Submit_Cmd']:
        val = str(row.get(col, '')).strip()
        if '/' in val:
            return val.split()[0]
    # ���ף���� Cmd �ǳ�����
    if row.get('Cmd', '').strip():
        return row['Cmd'].strip().split()[0]
    return 'unknown'

df['ProgramName'] = df['Cmd'].apply(extract_prog_name)
df['ProgramPath'] = df.apply(get_program_path, axis=1)

# ==== ����ProgramName ���� ====
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

# ==== ������ProgramName + ProgramPath �ֲ� ====
detail_table = (
    df.groupby(['ProgramName', 'ProgramPath'])
      .size()
      .reset_index(name='count')
      .sort_values(['ProgramName', 'count'], ascending=[True, False])
)
detail_table.to_csv(os.path.join(OUTPUT_DIR, "ProgramName_path_detail.csv"), index=False)

# ==== ��־��� ====
print(f" finish")
print(f" ProgramName total: {df['ProgramName'].nunique()}")
print("  ProgramName empty:", (df['ProgramName'] == '').sum())
print(f" summary save to: {os.path.join(OUTPUT_DIR, 'ProgramName_summary.csv')}")
print(f" path detail save to: {os.path.join(OUTPUT_DIR, 'ProgramName_path_detail.csv')}")
