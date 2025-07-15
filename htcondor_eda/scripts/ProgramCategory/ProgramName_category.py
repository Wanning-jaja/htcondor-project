# -*- coding: utf-8 -*-

import pandas as pd
import os
import re

# ===== 路径配置 =====
MAPPING_CSV = "/home/master/wzheng/projects/htcondor_eda/scripts/ProgramCategory/program_name_mapping.csv"
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_step2/ProgramName_unique_distribution.csv"
OUTPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory/ProgramName_unified_grouped.csv"
DETAIL_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory/ProgramName_lowfreq_undefined_detail.csv"
LOWFREQ_ONLY_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory/ProgramName_lowfreq_only.csv"
OTHERS_ONLY_CSV = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory/ProgramName_others_only.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

LOW_COUNT_THRESHOLD = 300
MAX_NAMES_DISPLAY = 5
#MAX_LINE_LEN = 200

# ===== 加载 mapping 表，按顺序匹配 =====
mapping_df = pd.read_csv(MAPPING_CSV)

def get_category(name, mapping_df):
    if pd.isnull(name) or str(name).strip() == "":
        return 'undefined'
    for _, row in mapping_df.iterrows():
        pattern = row['ProgramNamePattern']
        if re.match(pattern, str(name)):
            return row['ProgramCategory']
    return 'undefined'  # 未被 mapping 捕获的归入 undefined

#df = pd.read_csv(INPUT_CSV)
#df['ProgramCategory'] = df['ProgramName'].apply(lambda x: get_category(x, mapping_df))
df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
df.columns = df.columns.str.strip()
print("columns name:", df.columns.tolist())
print(df.head())

df['ProgramCategory'] = df['ProgramName'].apply(lambda x: get_category(x, mapping_df))

# ===== 聚合统计每个 ProgramName 的 count =====
stats = df.groupby(['ProgramCategory', 'ProgramName'], as_index=False)['count'].sum()

# ===== 分组归类 =====
main_rows, low_freq_rows, undefined_rows = [], [], []

for _, row in stats.iterrows():
    cat, name, cnt = row['ProgramCategory'], row['ProgramName'], row['count']
    if cat == 'undefined':
        undefined_rows.append({'ProgramCategory': 'undefined', 'ProgramName': name, 'count': cnt})
    elif cnt <= LOW_COUNT_THRESHOLD:
        low_freq_rows.append({'ProgramCategory': 'low_freq', 'ProgramName': name, 'count': cnt})
    else:
        main_rows.append({'ProgramCategory': cat, 'ProgramName': name, 'count': cnt})

#main_df = pd.DataFrame(main_rows)
#low_freq_df = pd.DataFrame(low_freq_rows).sort_values('count', ascending=False)
#undefined_df = pd.DataFrame(undefined_rows).sort_values('count', ascending=False)
main_df = pd.DataFrame(main_rows)
low_freq_df = pd.DataFrame(low_freq_rows)
undefined_df = pd.DataFrame(undefined_rows)

if not low_freq_df.empty:
    low_freq_df = low_freq_df.sort_values('count', ascending=False)
if not undefined_df.empty:
    undefined_df = undefined_df.sort_values('count', ascending=False)

# ===== 汇总聚合（主分组 + 附加两个） =====
def format_names(names, maxn=5, maxlen=None):
    names = sorted(set(str(n) for n in names))
    result = names[:maxn]
    tail = ""
    if len(names) > maxn:
        tail = f"... (total {len(names)})"
    return ";".join(result) + tail

def names_agg(df_):
    if df_.empty:
        return pd.DataFrame(columns=["ProgramCategory", "ProgramNames", "count"])
    return df_.groupby('ProgramCategory').agg(
        ProgramNames=('ProgramName', lambda x: format_names(x, MAX_NAMES_DISPLAY)),
        count=('count','sum')
    ).reset_index().sort_values('count', ascending=False)

grouped_main = names_agg(main_df)
grouped_lowfreq = names_agg(low_freq_df)
grouped_undefined = names_agg(undefined_df)

# ===== 写入主分组汇总表 =====
all_grouped = pd.concat([grouped_main, grouped_lowfreq, grouped_undefined], ignore_index=True)
all_grouped.to_csv(OUTPUT_CSV, index=False)
print(f"main save to: {OUTPUT_CSV}")

# ===== 写入明细（合并） =====
#detail_df = pd.concat([
#    low_freq_df.assign(GroupType='low_freq'),
#    undefined_df.assign(GroupType='others')
#], ignore_index=True)
#detail_df[['GroupType', 'ProgramCategory', 'ProgramName', 'count']].to_csv(DETAIL_CSV, index=False)
#print(f"low_freq + others detial to: {DETAIL_CSV}")

# ===== 附表：仅低频输出 =====
low_freq_df[['ProgramName', 'count']].to_csv(LOWFREQ_ONLY_CSV, index=False)
print(f"low_freq to: {LOWFREQ_ONLY_CSV}")

# ===== 附表：仅others输出（原始 ProgramCategory 为 others 的） =====
others_df = df[df['ProgramCategory'] == 'others'].sort_values('count', ascending=False)
others_df[['ProgramName', 'count']].to_csv(OTHERS_ONLY_CSV, index=False)
print(f"others to: {OTHERS_ONLY_CSV}")
print(f"Total main categories: {grouped_main.shape[0]}")
print(f"Total low_freq entries: {low_freq_df.shape[0]}")
print(f"Total others entries: {others_df.shape[0]}")

