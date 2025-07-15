# -*- coding: utf-8 -*-  #

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

program_files = {
    "runpilot2-wrapper.sh": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/runpilot2-wrapper.sh_records.csv",
    "superstar": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/superstar_records.csv",
    "star_errsolver.sh": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/star_errsolver_records.csv"
}

numeric_fields = [
    "RequestCpus", "RequestMemory", "RemoteWallClockTime",
    "CumulativeRemoteUserCpu", "CumulativeRemoteSysCpu",
    "ResidentSetSize_RAW", "ImageSize_RAW"
]

categorical_fields = ["ProgramPath", "Owner", "OwnerGroup"]

output_dir = "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/runtime_correlations"
os.makedirs(output_dir, exist_ok=True)

for pname, path in program_files.items():
    df = pd.read_csv(path)
    
    # 处理数值字段
    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["RemoteWallClockTime"])

    # === 数值字段：相关系数 + 散点图 ===
    corr = df[numeric_fields].corr(method='pearson')["RemoteWallClockTime"].sort_values(ascending=False)
    corr.to_csv(os.path.join(output_dir, f"{pname}_numeric_correlations.csv"))

    for field in numeric_fields:
        if field != "RemoteWallClockTime":
            plt.figure()
            sns.scatterplot(data=df, x=field, y="RemoteWallClockTime", alpha=0.3)
            plt.title(f"{pname}: {field} vs. RemoteWallClockTime")
            plt.savefig(os.path.join(output_dir, f"{pname}_scatter_{field}.png"))
            plt.close()

    # === 类别字段：平均运行时间 + 箱线图 ===
    for cat_field in categorical_fields:
            # === 类别字段过滤空值（仅针对 Owner / OwnerGroup）===
        if cat_field in ["Owner", "OwnerGroup"]:
           df = df[df[cat_field].notna() & (df[cat_field].str.strip() != "")]

        agg = (
            df.groupby(cat_field)["RemoteWallClockTime"]
              .agg(["count", "mean", "max"])
              .sort_values("count", ascending=False)
              .head(20)
        )
        agg.to_csv(os.path.join(output_dir, f"{pname}_catstat_{cat_field}.csv"))

        # 只画前20类的箱线图
        top_vals = agg.index.tolist()
        sub_df = df[df[cat_field].isin(top_vals)]

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=sub_df, x=cat_field, y="RemoteWallClockTime")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{pname}: RemoteWallClockTime by {cat_field} (Top 20)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pname}_box_{cat_field}.png"))
        plt.close()

print(" Analysis of the Numeric + Category fields in relation to runtime has been completed.")
