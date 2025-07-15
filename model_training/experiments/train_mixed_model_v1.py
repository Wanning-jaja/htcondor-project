# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# === 路径配置 ===
DATA_DIR = "/home/master/wzheng/projects/model_training/data"
MODEL_DIR = "/home/master/wzheng/projects/model_training/models"
REPORT_DIR = "/home/master/wzheng/projects/model_training/reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TOPN_JSON = os.path.join(DATA_DIR, "top44_programid_list.json")

# === 加载数据 ===
train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

# === 加载 Top-N ProgramID 列表 ===
with open(TOPN_JSON, "r") as f:
    top_program_ids = set(json.load(f))

# === 准备输出结果 ===
summary = []

# === 特征列和目标列 ===
feature_cols = [
    col for col in train_df.columns 
    if col not in ['RemoteWallClockTime', 'SubmitTime', 'ProgramID_encoded']
]
target_col = 'RemoteWallClockTime'

# === 对每个 Top-N 的 ProgramID 单独建模 ===
for prog_id in top_program_ids:
    sub_train = train_df[train_df['ProgramID_encoded'] == prog_id]
    sub_val = val_df[val_df['ProgramID_encoded'] == prog_id]

    if len(sub_train) < 10 or len(sub_val) < 10:
        continue  # 跳过过小的训练或验证集

    model = XGBRegressor(n_jobs=-1, random_state=42)
    model.fit(sub_train[feature_cols], sub_train[target_col])

    val_pred = model.predict(sub_val[feature_cols])
    #rmse = mean_squared_error(sub_val[target_col], val_pred, squared=False)
    mse = mean_squared_error(sub_val[target_col], val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(sub_val[target_col], val_pred)

    joblib.dump(model, os.path.join(MODEL_DIR, f"top{prog_id}_model.pkl"))
    summary.append({
        "ProgramID_encoded": prog_id,
        "Model": f"top{prog_id}_model.pkl",
        "TrainSize": len(sub_train),
        "ValSize": len(sub_val),
        "RMSE": rmse,
        "MAE": mae
    })

# === 建模“others”组 ===
train_others = train_df[~train_df['ProgramID_encoded'].isin(top_program_ids)]
val_others = val_df[~val_df['ProgramID_encoded'].isin(top_program_ids)]

if len(train_others) > 10 and len(val_others) > 10:
    model = XGBRegressor(n_jobs=-1, random_state=42)
    model.fit(train_others[feature_cols], train_others[target_col])

    val_pred = model.predict(val_others[feature_cols])
    #rmse = mean_squared_error(val_others[target_col], val_pred, squared=False)
    mse = mean_squared_error(val_others[target_col], val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(val_others[target_col], val_pred)

    joblib.dump(model, os.path.join(MODEL_DIR, "others_model.pkl"))
    summary.append({
        "ProgramID_encoded": "others",
        "Model": "others_model.pkl",
        "TrainSize": len(train_others),
        "ValSize": len(val_others),
        "RMSE": rmse,
        "MAE": mae
    })

# === 输出总结报告 ===
pd.DataFrame(summary).to_csv(os.path.join(REPORT_DIR, "evaluation_summary.csv"), index=False)
print("Mixed model training completed, results saved in:", REPORT_DIR)
