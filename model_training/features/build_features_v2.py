# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib  # ? ���������ڱ��������

# === ·������ ===
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/model_features_v1.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/data/"
ENCODER_DIR = os.path.join(OUTPUT_DIR, "encoders")  # ? �������������Ŀ¼
os.makedirs(ENCODER_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === �������� ===
df = pd.read_csv(INPUT_CSV)

# === Label Encoding ����������� ===
label_cols = ['Owner', 'OwnerGroup', 'ProgramID', 'ProgramName']
for col in label_cols:
    df[col] = df[col].fillna("missing")
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    
    # ? ���������Ϊ joblib �ļ�
    joblib.dump(le, os.path.join(ENCODER_DIR, f"{col}_encoder.joblib"))

# === ��ֵ�ֶ� log1p �任 ===
numeric_cols = [
    'RequestCpus', 'RequestMemory', 'RequestDisk',
    'ResidentSetSize_RAW', 'ImageSize_RAW',
    'JobCount'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df[col + "_log1p"] = np.log1p(df[col])

# === ʱ����Ŀ���ֶα��� ===
df['SubmitTime'] = pd.to_numeric(df['SubmitTime'], errors='coerce').astype('Int64')
df['RemoteWallClockTime'] = pd.to_numeric(df['RemoteWallClockTime'], errors='coerce')

# === ������������� ===
model_df = df[
    [col + "_encoded" for col in label_cols] +
    [col + "_log1p" for col in numeric_cols] +
    ['SubmitTime', 'RemoteWallClockTime']
].copy()

output_path = os.path.join(OUTPUT_DIR, "model_features_v2.csv")
model_df.to_csv(output_path, index=False)

print("? Feature engineering v2 completed:", output_path)
print("? Label encoders saved to:", ENCODER_DIR)
