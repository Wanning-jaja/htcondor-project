import os
import pandas as pd
import matplotlib.pyplot as plt

# 配置
INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes.csv"
OUTPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/field_analysis_results"
FIELDS = ["Cmd", "Owner", "OwnerGroup", "Arguments", "RemoteWallClockTime"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
summary_records = []

for field in FIELDS:
    field_data = df[field] if field in df.columns else pd.Series(dtype=str)
    non_empty = field_data[field_data != ""]
    total = len(field_data)
    missing = total - len(non_empty)
    unique_count = non_empty.nunique()
    top20 = non_empty.value_counts().head(20)

    # 保存 top20
    top20_path = os.path.join(OUTPUT_DIR, f"top20_{field}.csv")
    top20.to_csv(top20_path, header=["Count"])

    # 判断是否为数值字段
    try:
        numeric_series = pd.to_numeric(non_empty, errors="coerce").dropna()
        if len(numeric_series) > 100:
            plt.figure(figsize=(8, 4))
            numeric_series.hist(bins=50)
            plt.title(f"{field} - Numeric Distribution")
            plt.xlabel(field)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{field}.png"))
            plt.close()
    except:
        pass

    summary_records.append({
        "field": field,
        "total": total,
        "missing": missing,
        "missing_percent": round(missing / total * 100, 2),
        "unique_values": unique_count
    })

# 保存字段汇总统计
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "field_summary.csv"), index=False)
print("✅ Field analysis completed.")
