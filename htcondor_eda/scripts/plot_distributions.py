# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 配置路径 ====
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean_filtered.csv" 
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/distribution_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 加载数据 ====
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
numeric_fields = [
    'RemoteWallClockTime', 'RequestMemory', 'RequestCpus', 'RequestDisk',
    'ResidentSetSize_RAW', 'ImageSize_RAW', 'CumulativeRemoteUserCpu',
    'CumulativeRemoteSysCpu', 'CumulativeSuspensionTime'
]

# ==== 字段转为数值 ====
for field in numeric_fields:
    df[field] = pd.to_numeric(df[field], errors='coerce')

# ==== 绘图函数 ====
def plot_field(field, data):
    series = data[field].dropna()
    if series.empty:
        return
    log_series = np.log1p(series)

    # 原始直方图
    plt.figure()
    sns.histplot(series, bins=100, kde=False)
    plt.title(f"{field} Distribution")
    plt.xlabel(field)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{field}_hist.png"))
    plt.close()

    # 对数直方图
    plt.figure()
    sns.histplot(log_series, bins=100, kde=False)
    plt.title(f"{field} Log1p Distribution")
    plt.xlabel(f"log1p({field})")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{field}_log_hist.png"))
    plt.close()

    # 原始 boxplot
    plt.figure()
    sns.boxplot(x=series)
    plt.title(f"{field} Boxplot")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{field}_boxplot.png"))
    plt.close()

    # 对数 boxplot
    plt.figure()
    sns.boxplot(x=log_series)
    plt.title(f"{field} Log1p Boxplot")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{field}_log_boxplot.png"))
    plt.close()

# ==== 批量绘图 ====
for field in numeric_fields:
    print(f"Processing {field}...")
    plot_field(field, df)

# ==== 计算相关性热力图 ====
corr_df = df[numeric_fields].dropna()
corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

print("save to : ", OUTPUT_DIR)
