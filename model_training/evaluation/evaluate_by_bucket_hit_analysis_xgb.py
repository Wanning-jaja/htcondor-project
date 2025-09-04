# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# === 路径配置 ===
PREDICTION_CSV = "/home/master/wzheng/projects/model_training/reports/predictions_v3.2_xgb.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/evaluation/v3.2_xgb/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 自定义时间段 Bucket（单位：秒）===
bucket_edges = [0, 1800, 3600, 10800, 21600, 43200, np.inf]
bucket_labels = [
    "0-30min", "30min-1h", "1h-3h", "3h-6h",
    "6h-12h",  "12h+"
]

# === 加载预测数据 ===
df = pd.read_csv(PREDICTION_CSV)

# === 分配时间段 ===
df["true_bucket"] = pd.cut(df["true"], bins=bucket_edges, labels=bucket_labels, right=False)
df["pred_bucket"] = pd.cut(df["pred"], bins=bucket_edges, labels=bucket_labels, right=False)

# === 命中判断 ===
df["bucket_hit"] = df["true_bucket"] == df["pred_bucket"]

# === 错误方向分析 ===
# === 方向判断函数 ===
def compute_direction(row):
    if pd.isna(row["true_bucket"]) or pd.isna(row["pred_bucket"]):
        return "Invalid"
    if row["bucket_hit"]:
        return "Hit"
    true_idx = bucket_labels.index(row["true_bucket"])
    pred_idx = bucket_labels.index(row["pred_bucket"])
    return "Over" if pred_idx > true_idx else "Under"

df["direction"] = df.apply(compute_direction, axis=1)

# === 构造 bucket 上下边界映射（用于误差计算）===
bucket_upper_bounds = dict(zip(bucket_labels, bucket_edges[1:]))
bucket_lower_bounds = dict(zip(bucket_labels, bucket_edges[:-1]))

# === 误差幅度函数 ===
def compute_error(row):
    if row["direction"] == "Hit":
        return 0
    if row["direction"] == "Invalid":
        return np.nan
    if row["direction"] == "Over":
        return row["pred"] - bucket_upper_bounds[row["true_bucket"]]
    else:  # Under
        return bucket_lower_bounds[row["true_bucket"]] - row["pred"]

df["deviation_seconds"] = df.apply(compute_error, axis=1)


# === 汇总统计 ===
summary = df["direction"].value_counts().reindex(["Hit", "Under", "Over"]).fillna(0).astype(int)
summary.name = "count"
summary.to_csv(os.path.join(OUTPUT_DIR, "bucket_hit_summary.csv"))
print("Bucket hit statistics:")
print(summary)

# === 平均偏差幅度统计 ===
avg_deviation = df[df["direction"] != "Hit"].groupby("direction")["deviation_seconds"].mean().round(2)
avg_deviation.to_csv(os.path.join(OUTPUT_DIR, "bucket_deviation_mean.csv"))
print("\nAverage deviation in error direction (seconds):")
print(avg_deviation)

# === 图表输出 ===
plt.figure()
ax = summary.plot(kind="bar", color=["green", "red", "orange"])
plt.ylabel("Job Count")
plt.title("Bucket Prediction Result (Hit / Under / Over)")

# === 在柱子上方添加数值标签 ===
for i, val in enumerate(summary):
    ax.text(i, val + max(summary) * 0.01, str(val), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bucket_hit_result.png"))
plt.close()


plt.figure()
df[df["direction"] != "Hit"].boxplot(column="deviation_seconds", by="direction")
plt.title("Deviation by Error Direction")
plt.ylabel("Deviation (seconds)")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bucket_deviation_boxplot.png"))
plt.close()

print("\nThe charts and results have been saved to:", OUTPUT_DIR)
