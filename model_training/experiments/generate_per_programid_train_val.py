# -*- coding: utf-8 -*-
import pandas as pd
import json
import os

# === 配置路径 ===
FEATURES_CSV = "/home/master/wzheng/projects/model_training/data/model_features_v2.csv"
TOPN_JSON = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/data/topN_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载数据和 TopN 列表 ===
df = pd.read_csv(FEATURES_CSV)
with open(TOPN_JSON, 'r') as f:
    top_programids = json.load(f)

# === 遍历每个 TopN ProgramID，按 SubmitTime 分割 ===
for pid in top_programids:
    df_pid = df[df['ProgramID_encoded'] == pid].copy()
    df_pid = df_pid.sort_values(by='SubmitTime')

    # 划分比例（80% 训练，20% 验证）
    split_index = int(len(df_pid) * 0.8)
    train_df = df_pid.iloc[:split_index].copy()
    val_df = df_pid.iloc[split_index:].copy()

    # 保存结果
    train_path = os.path.join(OUTPUT_DIR, f"train_top{pid}.csv")
    val_path = os.path.join(OUTPUT_DIR, f"val_top{pid}.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Top {pid}: Train={len(train_df)} Val={len(val_df)}")

print("Time-sliced completion of all TopN ProgramIDs.")
