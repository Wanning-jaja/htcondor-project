# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# 输入配置
program_files = {
    "runpilot2-wrapper.sh": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/runpilot2-wrapper.sh_records.csv",
    "superstar": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/superstar_records.csv",
    "star_errsolver.sh": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/star_errsolver_records.csv"
}
output_dir = "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/runtime_distribution"
os.makedirs(output_dir, exist_ok=True)

for pname, path in program_files.items():
    try:
        # 读取数据并处理空值
        df = pd.read_csv(path)
        
        # 转换RemoteWallClockTime为数值，空值转为NaN
        df["RemoteWallClockTime"] = pd.to_numeric(df["RemoteWallClockTime"], errors='coerce')
        
        # 过滤掉NaN值
        df = df[df["RemoteWallClockTime"].notna()].copy()
        
        if len(df) == 0:
            print(f" {path} no valid RemoteWallClockTime, skip")
            continue
            
        # 生成直方图（对数刻度更清晰）
        plt.figure(figsize=(10, 6))
        sns.histplot(np.log1p(df["RemoteWallClockTime"]), bins=50, kde=True)
        plt.title(f"{pname} - Log(WallClockTime) Distribution")
        plt.xlabel("Log(RemoteWallClockTime + 1) (seconds)")
        plt.ylabel("Job Count")
        plt.savefig(os.path.join(output_dir, f"{pname}_wallclock_log_hist.png"))
        plt.close()

        # 生成箱线图（原始值）
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df["RemoteWallClockTime"])
        plt.title(f"{pname} - WallClockTime Boxplot")
        plt.xlabel("RemoteWallClockTime (seconds)")
        plt.savefig(os.path.join(output_dir, f"{pname}_wallclock_box.png"))
        plt.close()
        
        # 保存有效数据量
        print(f" {pname}: {len(df)} ")
        
    except Exception as e:
        print(f" deal {path} get error: {str(e)}")
        continue

print("runtime_correlations finished")

