# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import os

# === 配置路径 ===
ENCODER_PATH = "/home/master/wzheng/projects/model_training/data/encoders/ProgramID_encoder.joblib"
OUTPUT_PATH = "/home/master/wzheng/projects/model_training/data/ProgramID_encoding_map.csv"

# === 加载编码器 ===
le = joblib.load(ENCODER_PATH)

# === 构建编码映射表 ===
df_mapping = pd.DataFrame({
    "ProgramID_str": le.classes_,
    "ProgramID_encoded": le.transform(le.classes_)
})

# === 保存为 CSV 文件 ===
df_mapping.to_csv(OUTPUT_PATH, index=False)
print(f"ProgramID_encoding_map save to : {OUTPUT_PATH}")
