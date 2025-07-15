# -*- coding: utf-8 -*-

# split_dataset_by_time.py
import pandas as pd
import os

# === ����·�� ===
DATA_PATH = "/home/master/wzheng/projects/model_training/data/model_features_v2.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/data/"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
VAL_PATH = os.path.join(OUTPUT_DIR, "val.csv")

# === �������� ===
df = pd.read_csv(DATA_PATH)

# === �� SubmitTime ����������ʷ������ǰ�� ===
df_sorted = df.sort_values(by="SubmitTime").reset_index(drop=True)

# === ���ֱ�����80%ѵ������20%��֤���� ===
split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx]
val_df = df_sorted.iloc[split_idx:]

# === ������� ===
train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)

print(f"split_dataset_by_time finished : training :{len(train_df)} , Validation :{len(val_df)} ")
print(f"save to :\n training data :{TRAIN_PATH}\n validation data :{VAL_PATH}")
