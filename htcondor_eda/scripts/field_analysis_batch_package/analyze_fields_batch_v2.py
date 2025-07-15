
import os
import pandas as pd

# 配置路径
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 实际分析字段列表
FIELDS = ["Cmd", "Owner", "OwnerGroup", "Arguments_merged", "RemoteWallClockTime"]

# 加载数据
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

summary_records = []

for field in FIELDS:
    if field not in df.columns:
        print(f"⚠️ 字段不存在：{field}")
        continue

    field_data = df[field].astype(str)
    total = len(field_data)
    empty_count = (field_data == "").sum()
    undefined_count = (field_data == "undefined").sum()
    unique_count = field_data[field_data != ""].nunique()
    top10 = field_data.value_counts().head(10)

    # 保存 Top10
    top10.to_csv(os.path.join(OUTPUT_DIR, f"top10_{field}.csv"), header=["Count"])

    # 若为数值字段，尝试计算统计分布
    numeric_summary = {}
    try:
        numeric_series = pd.to_numeric(field_data, errors="coerce").dropna()
        if len(numeric_series) > 0:
            numeric_summary = {
                "min": numeric_series.min(),
                "max": numeric_series.max(),
                "mean": numeric_series.mean(),
                "median": numeric_series.median(),
                "p25": numeric_series.quantile(0.25),
                "p75": numeric_series.quantile(0.75)
            }
    except:
        pass

    summary_records.append({
        "field": field,
        "total": total,
        "empty": empty_count,
        "undefined": undefined_count,
        "unique_values": unique_count,
        **numeric_summary
    })

# 保存字段汇总统计
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "field_quality_overview.csv"), index=False)

print("✅ Field quality analysis completed.")
