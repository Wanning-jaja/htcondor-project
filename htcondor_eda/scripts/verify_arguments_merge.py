
import os
import pandas as pd

INPUT_CSV = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes2.csv"
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# 统计 Arguments == "undefined"
mask_args_undef = df["Arguments"].astype(str).str.strip() == "undefined"
count_undefined_arguments = mask_args_undef.sum()

# 检查是否存在 Args 字段
has_args_field = "Args" in df.columns
count_args_non_empty = 0
sample_non_empty_args = pd.DataFrame()

if has_args_field:
    args_series = df.loc[mask_args_undef, "Args"].astype(str)
    count_args_non_empty = (args_series.str.strip() != "").sum()
    sample_non_empty_args = df.loc[mask_args_undef & (args_series.str.strip() != ""), ["Arguments", "Args", "Cmd"]].head(20)
else:
    print("❌ 警告：数据中不包含 'Args' 字段，可能字段名拼写不一致或未导出。")

print("🧪 字段合并验证结果：")
print(f"➡️ Arguments == 'undefined' 的记录数: {count_undefined_arguments}")
if has_args_field:
    print(f"➡️ 其中 Args 非空的记录数: {count_args_non_empty}")
    print("\n🔍 示例记录（Arguments='undefined' 且 Args 有值）：")
    print(sample_non_empty_args.to_string(index=False))
