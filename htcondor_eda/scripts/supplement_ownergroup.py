# -*- coding: utf-8 -*-
import pandas as pd
import os

# �������·������
INPUT_PATH = "/home/master/wzheng/projects/htcondor_eda/results/no_ownergroup_merged_all_nodes_clean.csv"
OUTPUT_PATH = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean_with_ownergroup.csv"

# ��ȡ����
df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)

# ��ʽ�����µ� OwnerGroup_final ��
df['OwnerGroup_final'] = df['OwnerGroup']

# ��ȫ�߼����� OwnerGroup �ǿջ� undefined��ʹ�� x509UserProxyVOName
mask = df['OwnerGroup_final'].str.strip().isin(["", "undefined"])
df.loc[mask, 'OwnerGroup_final'] = df.loc[mask, 'x509UserProxyVOName']

# �����ֶΣ�ȥ��ԭ OwnerGroup��������Ϊ��׼��
df.drop(columns=['OwnerGroup'], inplace=True)
df.rename(columns={'OwnerGroup_final': 'OwnerGroup'}, inplace=True)

# ������
df.to_csv(OUTPUT_PATH, index=False)
print(f"OwnerGroup add finish save to : {OUTPUT_PATH}")
