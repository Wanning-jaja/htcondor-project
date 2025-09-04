# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os

# �����������·��
INPUT_CSV = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/evaluation/ownergroup_by_duration_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ��������
df = pd.read_csv(INPUT_CSV, low_memory=False)

# ����ʱ�����䣨��λ���룩�밲ȫ�ı�ǩ����
buckets = [
    (0, 600, "0_10min"),
    (600, 1800, "10_30min"),
    (1800, 3600, "30_60min"),
    (3600, 7200, "60_120min"),
    (7200, 14400, "120_240min")
]

# ����ÿ��ʱ��β���ͼ
for low, high, label in buckets:
    # ɸѡ��ǰ�����ڵ�����
    df_bucket = df[(df["RemoteWallClockTime"] > low) & (df["RemoteWallClockTime"] <= high)]

    # �����ǰ����û�����ݣ�����
    if df_bucket.empty:
        continue

    # ͳ�� OwnerGroup �����������������У�
    counts = df_bucket["OwnerGroup"].value_counts().sort_values(ascending=False)

    # �����µĻ�������������󣬱��⹲��״̬
    fig, ax = plt.subplots(figsize=(12, 6))

    # ������״ͼ
    counts.plot(kind="bar", ax=ax)

    # ���ñ���ͱ�ǩ
    ax.set_title(f"OwnerGroup Count in Duration {label}")
    ax.set_ylabel("Job Count")
    ax.set_xlabel("OwnerGroup")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # ��ÿ�������Ϸ������ֵ��ǩ
    for i, value in enumerate(counts.values):
        ax.text(i, value + 100, str(value), ha='center', va='bottom', fontsize=8, rotation=90)

    # ����ͼ��
    output_path = os.path.join(OUTPUT_DIR, f"ownergroup_count_{label}.png")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
