# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ��������
program_files = {
    "runpilot2-wrapper.sh": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/runpilot2-wrapper.sh_records.csv",
    "superstar": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/superstar_records.csv",
    "star_errsolver.sh": "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/star_errsolver_records.csv"
}
output_dir = "/home/master/wzheng/projects/htcondor_eda/results/typical_programs_analysis/runtime_distribution"
os.makedirs(output_dir, exist_ok=True)

for pname, path in program_files.items():
    try:
        # ��ȡ���ݲ������ֵ
        df = pd.read_csv(path)
        
        # ת��RemoteWallClockTimeΪ��ֵ����ֵתΪNaN
        df["RemoteWallClockTime"] = pd.to_numeric(df["RemoteWallClockTime"], errors='coerce')
        
        # ���˵�NaNֵ
        df = df[df["RemoteWallClockTime"].notna()].copy()
        
        if len(df) == 0:
            print(f" {path} no valid RemoteWallClockTime, skip")
            continue
            
        # ����ֱ��ͼ�������̶ȸ�������
        plt.figure(figsize=(10, 6))
        sns.histplot(np.log1p(df["RemoteWallClockTime"]), bins=50, kde=True)
        plt.title(f"{pname} - Log(WallClockTime) Distribution")
        plt.xlabel("Log(RemoteWallClockTime + 1) (seconds)")
        plt.ylabel("Job Count")
        plt.savefig(os.path.join(output_dir, f"{pname}_wallclock_log_hist.png"))
        plt.close()

        # ��������ͼ��ԭʼֵ��
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df["RemoteWallClockTime"])
        plt.title(f"{pname} - WallClockTime Boxplot")
        plt.xlabel("RemoteWallClockTime (seconds)")
        plt.savefig(os.path.join(output_dir, f"{pname}_wallclock_box.png"))
        plt.close()
        
        # ������Ч������
        print(f" {pname}: {len(df)} ")
        
    except Exception as e:
        print(f" deal {path} get error: {str(e)}")
        continue

print("runtime_correlations finished")

