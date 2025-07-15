# -*- coding: utf-8 -*-
import pandas as pd

# === �ļ�·����ʹ��δ��ȫ�汾 ===
INPUT = "/home/master/wzheng/projects/htcondor_eda/results/no_ownergroup_merged_all_nodes_clean.csv"

# === �������� ===
df = pd.read_csv(INPUT, dtype=str, low_memory=False)

# === �����ֶ� ===
df['OwnerGroup'] = df['OwnerGroup'].astype(str).str.strip()
df['x509UserProxyVOName'] = df['x509UserProxyVOName'].astype(str).str.strip()

# === ��ӡ�����ֶε�Ψһֵ���� ===
print("\n All unique OwnerGroups (non-empty and non-undefined):")
owner_unique = sorted(set(df['OwnerGroup']) - {"", "undefined"})
print(owner_unique[:20], "total: ", len(owner_unique))

print("\n All Unique VO (x509UserProxyVOName)(non-empty and non-undefined):")
vo_unique = sorted(set(df['x509UserProxyVOName']) - {"", "undefined"})
print(vo_unique[:20], "total: ", len(vo_unique))

# === ����Ƿ��� job ͬʱ���� OwnerGroup �� VO�����ǿ�/�� undefined��===
df_both = df[
    (~df['OwnerGroup'].isin(["", "undefined"])) &
    (~df['x509UserProxyVOName'].isin(["", "undefined"]))
]

both_count = len(df_both)
total = len(df)
print(f"\n?? Jobs with BOTH OwnerGroup and VO present: {both_count} / {total} = {both_count/total:.2%}")

# === չʾǰ 10 ��ʵ��������� ===
print("\n?? Sample (first 10) of jobs with both OwnerGroup and VO:")
print(df_both[['x509UserProxyVOName', 'OwnerGroup']].drop_duplicates().head(10).to_string(index=False))

# === ���� VO �� OwnerGroup ӳ���ϵ ===
vo_to_group = df_both.groupby('x509UserProxyVOName')['OwnerGroup'].nunique()
multi_map_vo = vo_to_group[vo_to_group > 1]

print(f"\n A total of VOs are present (non-empty and valid): {vo_to_group.shape[0]}")
print(f"\n The number of VOs mapped to MULTIPLE OwnerGroups: {multi_map_vo.shape[0]}")

if multi_map_vo.empty:
    print("\n All VOs map uniquely to one OwnerGroup mapping is valid for supplementation.")
else:
    print("\n? The following VOs map to multiple OwnerGroups (first 10):")
    vo_samples = df_both[df_both['x509UserProxyVOName'].isin(multi_map_vo.index)]
    vo_samples = vo_samples[['x509UserProxyVOName', 'OwnerGroup']].drop_duplicates()
    print(vo_samples.head(10).to_string(index=False))
