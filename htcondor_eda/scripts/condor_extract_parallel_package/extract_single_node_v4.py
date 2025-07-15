
import os
import sys
import gzip
import csv

if len(sys.argv) != 2:
    print("Usage: python extract_single_node_v4.py <node_name>")
    sys.exit(1)

node = sys.argv[1]
BASE_DIR = "/home/master/wzheng/projects/htcondor_eda/data"
OUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/condor_cli_extracted"
FIELDS = ['ClusterId', 'ProcId', 'Owner', 'OwnerGroup', 'Cmd', 'SUBMIT_Cmd', 'Arguments', 'Args',
          'JobStatus', 'ExitCode', 'EnteredCurrentStatus', 'CompletionDate', 'JobStartDate',
          'RemoteWallClockTime', 'CumulativeRemoteUserCpu', 'CumulativeRemoteSysCpu',
          'CumulativeSuspensionTime', 'ResidentSetSize_RAW', 'ImageSize_RAW', 'RequestCpus',
          'RequestMemory', 'RequestDisk', 'NumJobStarts', 'JobRunCount', 'LastRemoteHost', 'Iwd',
          'SubmitHost', 'GlobalJobId', 'x509UserProxyVOName', 'MATCH_GLIDEIN_Site',
          'MATCH_GLIDEIN_ResourceName', 'MATCH_EXP_JOB_GLIDEIN_Entry_Name']

os.makedirs(OUT_DIR, exist_ok=True)
node_path = os.path.join(BASE_DIR, node)
if not os.path.isdir(node_path):
    print(f"❌ Node directory not found: {node_path}")
    sys.exit(1)

output_csv = os.path.join(OUT_DIR, f"parsed_{node}.csv")
log_file = output_csv.replace(".csv", "_log.txt")
record_count = 0
short_count = 0

with open(output_csv, "w", encoding="utf-8", newline="") as fout, open(log_file, "w") as logf:
    writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
    writer.writerow(FIELDS)

    for fname in sorted(os.listdir(node_path)):
        if not fname.endswith(".gz"):
            continue
        gz_path = os.path.join(node_path, fname)
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    parts = line.strip().split("|")
                    if len(parts) < 3:
                        continue  # 空行或明显错误的跳过
                    row = [parts[i].strip() if i < len(parts) else "" for i in range(len(FIELDS))]
                    writer.writerow(row)
                    record_count += 1
                    if len(parts) < len(FIELDS):
                        short_count += 1
        except Exception as e:
            logf.write(f"❌ Error reading {fname}: {e}\n")

    logf.write(f"✅ Total records written: {record_count}\n")
    logf.write(f"⚠️ Records with fewer fields: {short_count}\n")
