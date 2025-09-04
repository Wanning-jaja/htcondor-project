# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === ���� ===
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/reports/ownergroup_runtime_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

selected_groups = [
    "magic", "atlas", "cta", "ciematdune",
    "euclid", "gcf", "magnesia", "paus", "deepdet"
]

# === �������� ===
df = pd.read_csv(INPUT_CSV, low_memory=False)
df["RemoteWallClockTime"] = pd.to_numeric(df["RemoteWallClockTime"], errors="coerce") / 60  # ��ת����
df = df[df["OwnerGroup"].isin(selected_groups)].copy()

# === ȥ������ʱ��Ϊ����ȱʧ�ļ�¼
df = df[df["RemoteWallClockTime"] > 0]
   
    # === ѭ��ÿ�� OwnerGroup ������ͼ
for group in selected_groups:
    df_group = df[df["OwnerGroup"] == group].copy()
    # ��ѡ��ȥ�� top 1% ����ֵ
    q99 = df_group["RemoteWallClockTime"].quantile(0.99)
    df_group = df_group[df_group["RemoteWallClockTime"] <= q99]

    # ���㶯̬ Y �᷶Χ
    ymin = df_group["RemoteWallClockTime"].min()
    ymax = df_group["RemoteWallClockTime"].max()

    plt.figure(figsize=(6, 5))
    sns.boxplot(x="OwnerGroup", y="RemoteWallClockTime", data=df_group, width=0.4, color='lightblue')
    sns.stripplot(x="OwnerGroup", y="RemoteWallClockTime", data=df_group, 
                  size=3, alpha=0.3, color='black', jitter=0.2)

    plt.title(f"{group}: Job Runtime Distribution")
    plt.ylabel("Runtime (minutes)")
    plt.xlabel("")
    plt.ylim(ymin * 0.9, ymax * 1.1)  # �������� 10% �ռ�
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # ����ͼ��
    plt.savefig(os.path.join(OUTPUT_DIR, f"{group}_runtime_boxplot.png"))
    plt.close()

print(f"save : {OUTPUT_DIR}")
