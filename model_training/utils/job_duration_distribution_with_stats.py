# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === 输入与输出路径 ===
INPUT = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/reports/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载数据 ===
df = pd.read_csv(INPUT, low_memory=False)
df['RemoteWallClockTime'] = pd.to_numeric(df['RemoteWallClockTime'], errors='coerce')
df = df[df['RemoteWallClockTime'].notna()]
df['WallTimeHours'] = df['RemoteWallClockTime'] / 3600.0

# === 定义分段 ===
bins = [0, 1/6, 0.5, 1, 2, 4, 6, 8, 12, 24, float("inf")]
labels = [
    "<10 mins", "<30 mins", "<1 hour", "<2 hours", "<4 hours",
    "<6 hours", "<8 hours", "<12 hours", "<24 hours", ">=24 hours"]
df['DurationBucket'] = pd.cut(df['WallTimeHours'], bins=bins, labels=labels, right=False)

# === 分段统计 + 百分比列
stats = df.groupby('DurationBucket', observed=False)['WallTimeHours'].agg(
    JobCount='count',
    MinTime='min',
    AvgTime='mean',
    MaxTime='max'
).reset_index()

total_jobs = stats['JobCount'].sum()
stats['JobCountPercent'] = (stats['JobCount'] / total_jobs * 100).round(2)
stats[['MinTime', 'AvgTime', 'MaxTime']] = stats[['MinTime', 'AvgTime', 'MaxTime']].round(4)

# === 输出表格
csv_out = os.path.join(OUTPUT_DIR, "runtime_distribution_stats.csv")
stats.to_csv(csv_out, index=False)
print(f"Runtime distribution statistics have been saved: {csv_out}")

# === 图 1：Job Count per DurationBucket
plt.figure(figsize=(10, 6))
bars = plt.bar(stats['DurationBucket'], stats['JobCount'], color='seagreen')
plt.title("Job Count per Runtime Interval")
plt.xlabel("Runtime Interval")
plt.ylabel("Job Count")
plt.xticks(rotation=45)
plt.grid(axis='y')
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height):,}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)
plt.tight_layout()
fig_out1 = os.path.join(OUTPUT_DIR, "avg_runtime_per_bucket.png")
plt.savefig(fig_out1)
plt.close()
print(f"Saved: {fig_out1}")

# === 图 2 & 3：分两部分画 Min / Avg / Max 柱状图
stats_1 = stats.iloc[:6].copy()
stats_2 = stats.iloc[6:].copy()

def plot_triplet_bar(stats_subset, title, filename):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(stats_subset))
    width = 0.25
    bars1 = plt.bar(x - width, stats_subset['MinTime'], width=width, label='MinTime', color='skyblue')
    bars2 = plt.bar(x, stats_subset['AvgTime'], width=width, label='AvgTime', color='mediumseagreen')
    bars3 = plt.bar(x + width, stats_subset['MaxTime'], width=width, label='MaxTime', color='lightcoral')
    plt.xticks(ticks=x, labels=stats_subset['DurationBucket'], rotation=45)
    plt.xlabel("Runtime Interval")
    plt.ylabel("Runtime (hours)")
    plt.title(title)
    plt.legend()
    plt.grid(axis='y')

    # 添加数值标注（保留4位小数）
    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    annotate_bars(bars1)
    annotate_bars(bars2)
    annotate_bars(bars3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

plot_triplet_bar(stats_1, "Min / Avg / Max Runtime (<4 hours)", "runtime_triplet_bar_short.png")
plot_triplet_bar(stats_2, "Min / Avg / Max Runtime (>=4 hours)", "runtime_triplet_bar_long.png")
