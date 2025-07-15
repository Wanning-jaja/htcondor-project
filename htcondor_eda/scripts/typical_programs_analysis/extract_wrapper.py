# -*- coding: utf-8 -*-  #

import os
import pandas as pd

INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/runpilot2-wrapper.sh_records.csv"
TARGET_NAME = "runpilot2-wrapper.sh"

# ----------------------------
# ���ݶ�ȡ
# ----------------------------
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# ----------------------------
# ��������
# ----------------------------
def extract_prog_name(cmd):
    """Extract program name"""
    if pd.isnull(cmd) or cmd.strip() == '':
        return ''
    path = cmd.strip().split()[0].rstrip('/')
    return os.path.basename(path)

def extract_program_path(row):
    """Extract program path"""
    for col in ['Cmd', 'SUBMIT_Cmd']:
        val = str(row.get(col, '')).strip()
        if '/' in val:
            return val.split()[0]
    return str(row.get('Cmd', '')).strip().split()[0] if row.get('Cmd', '').strip() else 'unknown'

# ----------------------------
# ���ݴ���
# ----------------------------
df['ProgramName'] = df['Cmd'].apply(extract_prog_name)
df['ProgramPath'] = df.apply(extract_program_path, axis=1)

# ȷ��Ŀ���ֶδ���
keep_cols = [
    "ProgramName", "ProgramPath", "Owner", "OwnerGroup",
    "RequestCpus", "RequestMemory", "RemoteWallClockTime",
    "CumulativeRemoteUserCpu", "CumulativeRemoteSysCpu",
    "ResidentSetSize_RAW", "ImageSize_RAW"
]
keep_cols = [col for col in keep_cols if col in df.columns]
if not keep_cols:
    raise ValueError(" can not find any fields in CSV ")

# ɸѡĿ�����
df_filtered = df[df["ProgramName"] == TARGET_NAME].copy()

if df_filtered.empty:
    print(f"can not find any {TARGET_NAME} record")
else:
    # ----------------------------
    # ��ֵ�ֶ�ת�������ȴ���
    # ----------------------------
    numeric_fields = [
        "RequestCpus", "RequestMemory", "RemoteWallClockTime",
        "CumulativeRemoteUserCpu", "CumulativeRemoteSysCpu",
        "ResidentSetSize_RAW", "ImageSize_RAW"
    ]
    for nf in numeric_fields:
        if nf in df_filtered.columns:
            original_na = df_filtered[nf].isna().sum()
            df_filtered[nf] = pd.to_numeric(df_filtered[nf], errors='coerce')
            new_na = df_filtered[nf].isna().sum()
            if new_na > original_na:
                print(f"{nf} has {new_na - original_na} records trans number failed")

    # ----------------------------
    # �ı��ֶ���ϴ + �Ƿ�ֵͳ��
    # ----------------------------
    text_fields = [col for col in keep_cols if col not in numeric_fields]
    null_values = ["", "undefined", "nan", "null", "none"]
    for col in text_fields:
        df_filtered[col] = df_filtered[col].astype(str).str.strip()
        null_like = df_filtered[col].str.lower().isin(null_values).sum()
        if null_like > 0:
            print(f" {col} has {null_like} ({null_values})")

    # ----------------------------
    # ������
    # ----------------------------
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_filtered[keep_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"runpilot2-wrapper.sh finish, total {len(df_filtered)} records,save to {OUTPUT_CSV}")
