import os
import pandas as pd
from datetime import datetime

# 路径配置
INPUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/condor_cli_extracted"
OUTPUT_FILE = "/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv"
LOG_FILE = OUTPUT_FILE.replace(".csv", "_merge_log.txt")

# 固定字段顺序（手动确保一致性）
FIELDS = [
    'ClusterId', 'ProcId', 'Owner', 'OwnerGroup', 'Cmd', 'SUBMIT_Cmd',
    'Arguments', 'Args', 'JobStatus', 'ExitCode', 'EnteredCurrentStatus',
    'CompletionDate', 'JobStartDate', 'RemoteWallClockTime',
    'CumulativeRemoteUserCpu', 'CumulativeRemoteSysCpu',
    'CumulativeSuspensionTime', 'ResidentSetSize_RAW', 'ImageSize_RAW',
    'RequestCpus', 'RequestMemory', 'RequestDisk', 'NumJobStarts',
    'JobRunCount', 'LastRemoteHost', 'Iwd', 'SubmitHost', 'GlobalJobId',
    'x509UserProxyVOName', 'MATCH_GLIDEIN_Site', 'MATCH_GLIDEIN_ResourceName',
    'MATCH_EXP_JOB_GLIDEIN_Entry_Name'
]

merged_dfs = []
log_lines = [f"[{datetime.now()}] Start merging all node CSVs..."]

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(INPUT_DIR, fname)
    try:
        df = pd.read_csv(fpath, dtype=str, keep_default_na=False)

        # 添加缺失字段
        for col in FIELDS:
            if col not in df.columns:
                df[col] = ""

        # 保证字段顺序统一
        df = df[FIELDS]

        # 全字段 strip，去掉首尾空白
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        # OwnerGroup 补全逻辑：用 x509UserProxyVOName 补 undefined 或空值
        mask_owner_undefined = df["OwnerGroup"].isin(["", "undefined"])
        mask_vo_valid = df["x509UserProxyVOName"].notna() & (df["x509UserProxyVOName"] != "")
        df.loc[mask_owner_undefined & mask_vo_valid, "OwnerGroup"] = df.loc[mask_owner_undefined & mask_vo_valid, "x509UserProxyVOName"]

        # 添加合并字段 Arguments_merged（先用 Args，再 fallback 到 Arguments）
        df["Arguments_merged"] = df["Args"]
        mask = df["Arguments_merged"].str.lower().isin(["", "undefined"])
        df.loc[mask, "Arguments_merged"] = df.loc[mask, "Arguments"]

        # 这里也 strip 一下 Arguments_merged，极致保险
        df["Arguments_merged"] = df["Arguments_merged"].astype(str).str.strip()


        # 添加 source_node 字段
        df["source_node"] = fname.replace("parsed_", "").replace(".csv", "")
        merged_dfs.append(df)
        log_lines.append(f"✅ {fname} parsed, records: {len(df)}")
    except Exception as e:
        log_lines.append(f"❌ Failed reading {fname}: {e}")

# 输出合并结果
if merged_dfs:
    full_df = pd.concat(merged_dfs, ignore_index=True)
    full_df.to_csv(OUTPUT_FILE, index=False)
    log_lines.append(f"✅ Merged total records: {len(full_df)}")
else:
    log_lines.append("❌ No files merged.")

with open(LOG_FILE, "w") as logf:
    logf.write("\n".join(log_lines))

print("✅ Merge complete. Output saved to:", OUTPUT_FILE)
