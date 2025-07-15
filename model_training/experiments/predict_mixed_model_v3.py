# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import json
from joblib import load
from sklearn.preprocessing import LabelEncoder

# === 路径配置 ===
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/val.csv"
MAPPING_CSV = "/home/master/wzheng/projects/model_training/utils/program_name_mapping.csv"
ENCODER_DIR = "/home/master/wzheng/projects/model_training/data/encoders"
MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v2"
TOPN_JSON = "/home/master/wzheng/projects/model_training/data/top44_programid_list.json"
OUTPUT_PREDICTION = "/home/master/wzheng/projects/model_training/reports/predictions_v3.csv"

# === 加载数据与辅助资源 ===
df = pd.read_csv(INPUT_CSV)
mapping_df = pd.read_csv(MAPPING_CSV)
with open(TOPN_JSON, "r") as f:
    top_programids = set(json.load(f))

# === 加载编码器 ===
#encoders = {}
#for col in ["Owner", "OwnerGroup", "ProgramID", "ProgramName"]:
#    le = LabelEncoder()
#    le.classes_ = np.load(os.path.join(ENCODER_DIR, f"{col}_classes.npy"), allow_pickle=True)
#    encoders[col] = le
# === 加载编码器 ===
encoders = {}
for col in ["Owner", "OwnerGroup", "ProgramID", "ProgramName"]:
    encoder_path = os.path.join(ENCODER_DIR, f"{col}_encoder.joblib")
    encoders[col] = load(encoder_path)


# === 重建 ProgramName 和 ProgramPath4（可选）===
df["ProgramName"] = df["ProgramName_encoded"].apply(lambda x: encoders["ProgramName"].inverse_transform([x])[0])
df["ProgramID_raw"] = df["ProgramID_encoded"].apply(lambda x: encoders["ProgramID"].inverse_transform([x])[0])

# === 预测 ===
results = []
for _, row in df.iterrows():
    pid = row["ProgramID_encoded"]
    #features = row.drop(["RemoteWallClockTime", "SubmitTime"]).values.reshape(1, -1)
    FEATURE_COLUMNS = [
    "Owner_encoded", "OwnerGroup_encoded", "ProgramID_encoded", "ProgramName_encoded",
    "RequestCpus_log1p", "RequestMemory_log1p", "RequestDisk_log1p",
    "ResidentSetSize_RAW_log1p", "ImageSize_RAW_log1p", "JobCount_log1p"
]
    features = row[FEATURE_COLUMNS].values.reshape(1, -1)

    model_name = f"xgb_model_pid{pid}.joblib" if pid in top_programids else "xgb_model_others.joblib"
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"Missing model files: {model_path}")
        continue

    model = load(model_path)
    pred = model.predict(features)[0]
    results.append({
        "true": row["RemoteWallClockTime"],
        "pred": pred,
        "ProgramID_encoded": pid
    })

# === 保存结果 ===
pd.DataFrame(results).to_csv(OUTPUT_PREDICTION, index=False)
print(f"Prediction is complete and results are saved to: {OUTPUT_PREDICTION}")
