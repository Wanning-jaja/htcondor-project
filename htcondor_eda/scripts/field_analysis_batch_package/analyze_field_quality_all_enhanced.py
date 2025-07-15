import os
import pandas as pd

# 路径按需修改
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
FIELDS = df.columns.tolist()

summary_records = []

for field in FIELDS:
    field_data = df[field].astype(str)
    total = len(field_data)
    undefined_count = (field_data == "undefined").sum()
    empty_count = (field_data == "").sum()
    defined_count = total - undefined_count - empty_count
    unique_count = field_data[field_data != ""].nunique()
    # Top 10原样输出
    top10 = field_data.value_counts().head(10)
    top10_df = top10.reset_index()
    top10_df.columns = ["value", "count"]
    top10_df.to_csv(os.path.join(OUTPUT_DIR, f"top10_{field}.csv"), index=False)

    summary_records.append({
        "field": field,
        "total": total,
        "undefined": undefined_count,
        "empty": empty_count,
        "defined": defined_count,
        "unique_values": unique_count
    })

# 总览表
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "field_quality_overview.csv"), index=False)

# Debug输出
print("全部字段：", FIELDS)
print("Arguments_merged 唯一值举例：", df['Arguments_merged'].unique()[:10])
print("Arguments_merged == 'undefined' 的数量：", (df['Arguments_merged'] == "undefined").sum())
print("字段分析完成，所有统计均基于原始内容，无任何加工。")
