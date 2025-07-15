# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from joblib import dump

# === 配置 ===
SPLIT_DIR = "/home/master/wzheng/projects/model_training/data/topN_splits"
TOPN_JSON = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v2"
REPORT_DIR = "/home/master/wzheng/projects/model_training/reports"
ALL_TRAIN = "/home/master/wzheng/projects/model_training/data/train.csv"
ALL_VAL = "/home/master/wzheng/projects/model_training/data/val.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# === 加载 TopN ProgramID 列表 ===
with open(TOPN_JSON, 'r') as f:
    top_programids = json.load(f)

results = []

# === 遍历每个 Top-N ProgramID 训练独立模型 ===
for pid in top_programids:
    print(f"\n?? Training ProgramID {pid}...")

    train_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
    val_path = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"skip ProgramID={pid}, file does not exist")
        continue

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    feature_cols = [col for col in df_train.columns if col not in ['RemoteWallClockTime', 'SubmitTime']]
    target_col = 'RemoteWallClockTime'

    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(df_train[feature_cols], df_train[target_col])

    val_pred = model.predict(df_val[feature_cols])
    #rmse = np.sqrt(mean_squared_error(df_val[target_col], val_pred))
    mse = mean_squared_error(df_val[target_col], val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df_val[target_col], val_pred)

    # 保存模型
    model_path = os.path.join(MODEL_DIR, f"xgb_model_pid{pid}.joblib")
    dump(model, model_path)

    # 保存结果记录
    results.append({
        "ProgramID_encoded": pid,
        "TrainSize": len(df_train),
        "ValSize": len(df_val),
        "RMSE": rmse,
        "MAE": mae
    })

    print(f" PID={pid} | RMSE={rmse:.2f} | MAE={mae:.2f}")

# === 训练 Others 模型 ===
print("\n?? Training fallback model for Others...")
df_all_train = pd.read_csv(ALL_TRAIN)
df_all_val = pd.read_csv(ALL_VAL)

# 筛选 ProgramID 不在 Top-N 中的数据
df_others_train = df_all_train[~df_all_train['ProgramID_encoded'].isin(top_programids)].copy()
df_others_val = df_all_val[~df_all_val['ProgramID_encoded'].isin(top_programids)].copy()

feature_cols = [col for col in df_others_train.columns if col not in ['RemoteWallClockTime', 'SubmitTime']]
target_col = 'RemoteWallClockTime'

model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(df_others_train[feature_cols], df_others_train[target_col])

val_pred = model.predict(df_others_val[feature_cols])
#rmse = np.sqrt(mean_squared_error(df_others_val[target_col], val_pred))
mse = mean_squared_error(df_others_val[target_col], val_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(df_others_val[target_col], val_pred)

model_path = os.path.join(MODEL_DIR, f"xgb_model_others.joblib")
dump(model, model_path)

results.append({
    "ProgramID_encoded": "Others",
    "TrainSize": len(df_others_train),
    "ValSize": len(df_others_val),
    "RMSE": rmse,
    "MAE": mae
})

print(f" Others | RMSE={rmse:.2f} | MAE={mae:.2f}")

# === 汇总评估结果 ===
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE")
report_path = os.path.join(REPORT_DIR, "v2_evaluation_summary.csv")
results_df.to_csv(report_path, index=False)
print("\n All model training is complete and the evaluation report is saved:", report_path)
