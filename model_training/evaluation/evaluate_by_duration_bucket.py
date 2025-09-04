# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# === ·������ ===
PREDICTION_FILE = "/home/master/wzheng/projects/model_training/reports/predictions_v3.1.3.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/model_training/evaluation/v5_duration_bucket_evaluation.csv"
OUTPUT_IMG = "/home/master/wzheng/projects/model_training/evaluation/v5_duration_bucket_evaluation.png"

# === ����Ԥ������ ===
df = pd.read_csv(PREDICTION_FILE)

# === ����ʱ��Σ���λ�룩 ===
buckets = [
    (0, 600, "<10 mins"),
    (600, 1800, "10-30 mins"),
    (1800, 3600, "30 mins - 1 hour"),
    (3600, 7200, "1-2 hours"),
    (7200, 14400, "2-4 hours"),
    (14400, 21600, "4-6 hours"),
    (21600, 28800, "6-8 hours"),
    (28800, 43200, "8-12 hours"),
    (43200, 86400, "12-24 hours"),
    (86400, float("inf"), ">=24 hours")
]

# === �ֶ����� ===
results = []
epsilon = 1e-8  # ��ֹ����0

for low, high, label in buckets:
    mask = (df["true"] >= low) & (df["true"] < high)
    df_bucket = df[mask]
    if len(df_bucket) == 0:
        continue
    true = df_bucket["true"].values
    pred = df_bucket["pred"].values
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + epsilon))) * 100
    results.append({
        "DurationBucket": label,
        "SampleSize": len(df_bucket),
        "MAE": mae,
        "MAPE(%)": mape
    })

# === ������Ϊ CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Segmented assessment completed and results saved to: {OUTPUT_CSV}")

# === ���ӻ����ɣ�MAE ��״ͼ + MAPE ����ͼ + ��׼�ߣ� ===
fig, ax1 = plt.subplots(figsize=(12, 6))

x = results_df["DurationBucket"]
x_pos = np.arange(len(x))
bar_width = 0.5

# ���᣺MAE ��״ͼ
ax1.set_xlabel("Duration Bucket")
ax1.set_ylabel("MAE (seconds)", color='steelblue')
bar1 = ax1.bar(x_pos, results_df["MAE"], width=bar_width, label='MAE', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x, rotation=30)

# ���᣺MAPE ����ͼ
ax2 = ax1.twinx()
ax2.set_ylabel("MAPE (%)", color='orange')
line2 = ax2.plot(x_pos, results_df["MAPE(%)"], color='orange', marker='o', label='MAPE')
ax2.tick_params(axis='y', labelcolor='orange')

# ��ӻ�׼�ߣ��ɵ���ֵ��
ax2.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='100% threshold')
ax2.axhline(y=200, color='gray', linestyle=':', linewidth=1, label='200% threshold')

# �����ͼ��
fig.tight_layout()
plt.title("Evaluation by Job Duration (MAE + MAPE Line)")
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
plt.grid(axis='y', linestyle='--', alpha=0.5)

# === ����ͼ�� ===
plt.savefig(OUTPUT_IMG)
print(f"The chart has been saved: {OUTPUT_IMG}")

# === ����ͼ�� ===
plt.savefig(OUTPUT_IMG)
print(f"The chart has been saved: {OUTPUT_IMG}")
