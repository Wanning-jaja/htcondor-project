# -*- coding: utf-8 -*-

import pandas as pd
import json

# === ���� ===
TRAIN_PATH = "/home/master/wzheng/projects/model_training/data/train.csv"
TOPN = 44  # ����Ըĳ� Top-30 �� Top-50 ��
OUTPUT_JSON = f"/home/master/wzheng/projects/model_training/data/top{TOPN}_programid_list.json"

# === ����ѵ���� ===
df = pd.read_csv(TRAIN_PATH)

# === ͳ�� ProgramID ����Ƶ�� ===
top_programs = df['ProgramID_encoded'].value_counts().nlargest(TOPN).index.tolist()

# === ����Ϊ JSON �ļ� ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(top_programs, f)

print(f"Top-{TOPN} ProgramID coding finished , save to : {OUTPUT_JSON}")
