# -*- coding: utf-8 -*-

import pandas as pd
import os

INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# 预处理：统计每个 OwnerGroup 下有多少唯一 Owner
group_owner_count = df.groupby('OwnerGroup')['Owner'].nunique().to_dict()
# 统计每个 Owner 下有多少唯一 OwnerGroup
owner_group_count = df.groupby('Owner')['OwnerGroup'].nunique().to_dict()

# Owner视角
def get_ownergroups(owner):
    ogs = df[df['Owner'] == owner]['OwnerGroup'].dropna().unique()
    return sorted(ogs)

def get_ownergroup_owner_counts(owner):
    ogs = get_ownergroups(owner)
    return [f"{og}({group_owner_count.get(og, 0)})" for og in ogs]

owner_stats = []
for owner, group in df.groupby('Owner'):
    owner_count = len(group)
    ownergroups = get_ownergroups(owner)
    ownergroups_str = ';'.join(ownergroups)
    og_owner_counts = ';'.join(get_ownergroup_owner_counts(owner))
    owner_stats.append([owner, owner_count, ownergroups_str, og_owner_counts])

owner_stats_df = pd.DataFrame(owner_stats, columns=["Owner", "Owner_count", "OwnerGroups", "OwnerGroup_OwnerCounts"])
owner_stats_df.to_csv(os.path.join(OUTPUT_DIR, "owner_stats.csv"), index=False)

# OwnerGroup视角
owner_count_in_group = df.groupby('Owner')['OwnerGroup'].nunique().to_dict()
group_owner_stats = []
for group, data in df.groupby('OwnerGroup'):
    group_count = len(data)
    owners = data['Owner'].dropna().unique()
    owners_str = ';'.join(sorted(owners))
    owner_counts_str = ';'.join([f"{owner}({owner_count_in_group.get(owner, 0)})" for owner in sorted(owners)])
    group_owner_stats.append([group, group_count, owners_str, owner_counts_str])

group_owner_stats_df = pd.DataFrame(group_owner_stats, columns=["OwnerGroup", "OwnerGroup_count", "Owners", "Owner_OwnerGroupCounts"])
group_owner_stats_df.to_csv(os.path.join(OUTPUT_DIR, "ownergroup_stats.csv"), index=False)
