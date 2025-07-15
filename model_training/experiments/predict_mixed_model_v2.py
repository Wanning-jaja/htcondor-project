# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
from joblib import load

# === 配置路径 ===
TOPN_JSON = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
VAL_CSV = "/home/master/wzheng/projects/model_training/data/val.csv"
MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v2"
OUTPUT_PATH = "/home/master/wzheng/projects/model_training/reports/predictions_v2.csv"

# === 加载验证集和 TopN 列表 ===
df_val = pd.read_csv(VAL_CSV)
with open(TOPN_JSON, 'r') as f:
    top_programids = set(json.load(f))

# === 初始化输出列表 ===
all_predictions = []

# === 遍历所有验证集记录 ===
for pid in df_val['ProgramID_encoded'].unique():
    df_subset = df_val[df_val['ProgramID_encoded'] == pid].copy()
    feature_cols = [col for col in df_subset.columns if col not in ['RemoteWallClockTime', 'SubmitTime']]

    if pid in top_programids:
        model_path = os.path.join(MODEL_DIR, f"xgb_model_pid{pid}.joblib")
    else:
        model_path = os.path.join(MODEL_DIR, "xgb_model_others.joblib")

    if not os.path.exists(model_path):
        print(f"model lose :{model_path}, skip ProgramID={pid}")
        continue

    model = load(model_path)
    preds = model.predict(df_subset[feature_cols])

    pred_df = pd.DataFrame({
        'ProgramID_encoded': pid,
        'true': df_subset['RemoteWallClockTime'].values,
        'pred': preds
    })
    all_predictions.append(pred_df)

# === 合并保存所有预测结果 ===
final_df = pd.concat(all_predictions, ignore_index=True)
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"predictions finished , saved to : {OUTPUT_PATH}")
