# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# === ·������ ===
PREDICTION_FILE = "/home/master/wzheng/projects/model_training/reports/predictions_v3.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/model_training/evaluation/v3_duration_bucket_evaluation.csv"
OUTPUT_IMG = "/home/master/wzheng/projects/model_training/evaluation/v3_duration_bucket_evaluation.png"

# === ����Ԥ������ ===
df = pd.read_csv(PREDICTION_FILE)

# === ����ʱ��Σ���λ�룩 ===
buckets = [
    (0, 3600, "0-1h"),
    (3600, 14400, "1-4h"),
    (14400, 21600, "4-6h"),
    (21600, 86400, "6-24h"),
    (86400, float('inf'), ">=24h")
]

# === �ֶ����� ===
results = []
for low, high, label in buckets:
    mask = (df["true"] > low) & (df["true"] <= high)
    df_bucket = df[mask]
    if len(df_bucket) == 0:
        continue
    rmse = np.sqrt(mean_squared_error(df_bucket["true"], df_bucket["pred"]))
    mae = mean_absolute_error(df_bucket["true"], df_bucket["pred"])
    results.append({
        "DurationBucket": label,
        "SampleSize": len(df_bucket),
        "RMSE": rmse,
        "MAE": mae
    })

# === ������Ϊ CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f" Segmented assessment completed and results saved to: {OUTPUT_CSV}")

# === ���ӻ����� ===
plt.figure(figsize=(10, 6))
x = results_df["DurationBucket"]
rmse_values = results_df["RMSE"]
mae_values = results_df["MAE"]

bar_width = 0.35
x_pos = np.arange(len(x))

plt.bar(x_pos - bar_width/2, rmse_values, width=bar_width, label='RMSE', color='steelblue')
plt.bar(x_pos + bar_width/2, mae_values, width=bar_width, label='MAE', color='orange')

plt.xticks(x_pos, x)
plt.xlabel("Duration Bucket")
plt.ylabel("Error")
plt.title("Evaluation by Job Duration")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# === ����ͼ�� ===
plt.savefig(OUTPUT_IMG)
print(f" The chart has been saved :{OUTPUT_IMG}")
