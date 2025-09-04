# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置输入输出路径
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/evaluation/ownergroup_by_duration_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据
df = pd.read_csv(INPUT_CSV, low_memory=False)

# 定义时间区间（单位：秒）与安全的标签命名
buckets = [
    (0, 600, "0_10min"),
    (600, 1800, "10_30min"),
    (1800, 3600, "30_60min"),
    (3600, 7200, "60_120min"),
    (7200, 14400, "120_240min")
]

# 遍历每个时间段并绘图
for low, high, label in buckets:
    # 筛选当前区间内的数据
    df_bucket = df[(df["RemoteWallClockTime"] > low) & (df["RemoteWallClockTime"] <= high)]

    # 如果当前区间没有数据，跳过
    if df_bucket.empty:
        continue

    # 统计 OwnerGroup 的数量（按降序排列）
    counts = df_bucket["OwnerGroup"].value_counts().sort_values(ascending=False)

    # 创建新的画布和坐标轴对象，避免共享状态
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    counts.plot(kind="bar", ax=ax)

    # 设置标题和标签
    ax.set_title(f"OwnerGroup Count in Duration {label}")
    ax.set_ylabel("Job Count")
    ax.set_xlabel("OwnerGroup")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 在每个柱子上方添加数值标签
    for i, value in enumerate(counts.values):
        ax.text(i, value + 100, str(value), ha='center', va='bottom', fontsize=8, rotation=90)

    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, f"ownergroup_count_{label}.png")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
