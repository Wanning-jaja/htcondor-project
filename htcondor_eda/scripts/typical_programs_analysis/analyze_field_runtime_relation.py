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
    
    # ������ֵ�ֶ�
    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["RemoteWallClockTime"])

    # === ��ֵ�ֶΣ����ϵ�� + ɢ��ͼ ===
    corr = df[numeric_fields].corr(method='pearson')["RemoteWallClockTime"].sort_values(ascending=False)
    corr.to_csv(os.path.join(output_dir, f"{pname}_numeric_correlations.csv"))

    for field in numeric_fields:
        if field != "RemoteWallClockTime":
            plt.figure()
            sns.scatterplot(data=df, x=field, y="RemoteWallClockTime", alpha=0.3)
            plt.title(f"{pname}: {field} vs. RemoteWallClockTime")
            plt.savefig(os.path.join(output_dir, f"{pname}_scatter_{field}.png"))
            plt.close()

    # === ����ֶΣ�ƽ������ʱ�� + ����ͼ ===
    for cat_field in categorical_fields:
            # === ����ֶι��˿�ֵ������� Owner / OwnerGroup��===
        if cat_field in ["Owner", "OwnerGroup"]:
           df = df[df[cat_field].notna() & (df[cat_field].str.strip() != "")]

        agg = (
            df.groupby(cat_field)["RemoteWallClockTime"]
              .agg(["count", "mean", "max"])
              .sort_values("count", ascending=False)
              .head(20)
        )
        agg.to_csv(os.path.join(output_dir, f"{pname}_catstat_{cat_field}.csv"))

        # ֻ��ǰ20�������ͼ
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
