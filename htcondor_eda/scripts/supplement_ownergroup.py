# -*- coding: utf-8 -*-
import pandas as pd
import os

# 输入输出路径配置
INPUT_PATH = "/home/master/wzheng/projects/htcondor_eda/results/no_ownergroup_merged_all_nodes_clean.csv"
OUTPUT_PATH = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean_with_ownergroup.csv"

# 读取数据
df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)

# 显式创建新的 OwnerGroup_final 列
df['OwnerGroup_final'] = df['OwnerGroup']

# 补全逻辑：若 OwnerGroup 是空或 undefined，使用 x509UserProxyVOName
mask = df['OwnerGroup_final'].str.strip().isin(["", "undefined"])
df.loc[mask, 'OwnerGroup_final'] = df.loc[mask, 'x509UserProxyVOName']

# 清理字段：去除原 OwnerGroup，重命名为标准列
df.drop(columns=['OwnerGroup'], inplace=True)
df.rename(columns={'OwnerGroup_final': 'OwnerGroup'}, inplace=True)

# 输出结果
df.to_csv(OUTPUT_PATH, index=False)
print(f"OwnerGroup add finish save to : {OUTPUT_PATH}")
