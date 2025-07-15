# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import os

# === ����·�� ===
ENCODER_PATH = "/home/master/wzheng/projects/model_training/data/encoders/ProgramID_encoder.joblib"
OUTPUT_PATH = "/home/master/wzheng/projects/model_training/data/ProgramID_encoding_map.csv"

# === ���ر����� ===
le = joblib.load(ENCODER_PATH)

# === ��������ӳ��� ===
df_mapping = pd.DataFrame({
    "ProgramID_str": le.classes_,
    "ProgramID_encoded": le.transform(le.classes_)
})

# === ����Ϊ CSV �ļ� ===
df_mapping.to_csv(OUTPUT_PATH, index=False)
print(f"ProgramID_encoding_map save to : {OUTPUT_PATH}")
