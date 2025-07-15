# -*- coding: utf-8 -*-

import pandas as pd
import json

# === 配置 ===
TRAIN_PATH = "/home/master/wzheng/projects/model_training/data/train.csv"
TOPN = 44  # 你可以改成 Top-30 或 Top-50 等
OUTPUT_JSON = f"/home/master/wzheng/projects/model_training/data/top{TOPN}_programid_list.json"

# === 加载训练集 ===
df = pd.read_csv(TRAIN_PATH)

# === 统计 ProgramID 编码频次 ===
top_programs = df['ProgramID_encoded'].value_counts().nlargest(TOPN).index.tolist()

# === 保存为 JSON 文件 ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(top_programs, f)

print(f"Top-{TOPN} ProgramID coding finished , save to : {OUTPUT_JSON}")
