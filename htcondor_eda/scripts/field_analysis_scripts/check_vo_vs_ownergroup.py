# -*- coding: utf-8 -*-
import pandas as pd

# === 原始未补全数据路径 ===
INPUT_PATH = "/home/master/wzheng/projects/htcondor_eda/results/no_ownergroup_merged_all_nodes_clean.csv"

# === 加载数据 ===
df = pd.read_csv(INPUT_PATH, dtype=str, low_memory=False)
df['OwnerGroup'] = df['OwnerGroup'].astype(str).str.strip()
df['x509UserProxyVOName'] = df['x509UserProxyVOName'].astype(str).str.strip()

# === 空值判断逻辑 ===
def is_empty(val):
    return val.lower() in ['', 'undefined']

# === 统计数量 ===
total = len(df)
ownergroup_missing = df['OwnerGroup'].apply(is_empty).sum()
vo_missing = df['x509UserProxyVOName'].apply(is_empty).sum()

# === 打印结果 ===
print(f"Total number of records : {total}")
print(f" Original OwnerGroup is missing (empty or undefined): {ownergroup_missing} / {total} = {ownergroup_missing/total:.2%}")
print(f" VO field x509UserProxyVOName missing count: {vo_missing} / {total} = {vo_missing/total:.2%}")

# === 分析 fallback 替代潜力 ===
fallback_available = df.apply(lambda row: is_empty(row['OwnerGroup']) and not is_empty(row['x509UserProxyVOName']), axis=1).sum()
print(f"Number of records available for VO completion (OwnerGroup is missing but VO has value):{fallback_available}")

