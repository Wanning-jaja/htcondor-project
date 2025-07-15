#!/usr/bin/env python3
import os
import sys
import gzip
import subprocess
import csv

if len(sys.argv) != 2:
    print("Usage: python extract_single_node_v3.py <node_name>")
    sys.exit(1)

node = sys.argv[1]
BASE_DIR = "/home/master/wzheng/projects/htcondor_eda/data"
OUT_DIR = "/home/master/wzheng/projects/htcondor_eda/results/condor_cli_extracted"
FIELDS = ['ClusterId', 'ProcId', 'Owner', 'OwnerGroup', 'Cmd', 'SUBMIT_Cmd', 'Arguments', 'Args', 'JobStatus', 'ExitCode', 'EnteredCurrentStatus', 'CompletionDate', 'JobStartDate', 'RemoteWallClockTime', 'CumulativeRemoteUserCpu', 'CumulativeRemoteSysCpu', 'CumulativeSuspensionTime', 'ResidentSetSize_RAW', 'ImageSize_RAW', 'RequestCpus', 'RequestMemory', 'RequestDisk', 'NumJobStarts', 'JobRunCount', 'LastRemoteHost', 'Iwd', 'SubmitHost', 'GlobalJobId', 'x509UserProxyVOName', 'MATCH_GLIDEIN_Site', 'MATCH_GLIDEIN_ResourceName', 'MATCH_EXP_JOB_GLIDEIN_Entry_Name']

node_path = os.path.join(BASE_DIR, node)
if not os.path.isdir(node_path):
    print(f"❌ Node directory not found: {node_path}")
    sys.exit(1)

out_csv = os.path.join(OUT_DIR, f"parsed_{node}.csv")
os.makedirs(OUT_DIR, exist_ok=True)

with open(out_csv, "w", encoding="utf-8", newline='') as f_out:
    writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
    writer.writerow(FIELDS)

    for file in sorted(os.listdir(node_path)):
        if not file.endswith(".gz"):
            continue
        gz_path = os.path.join(node_path, file)
        txt_path = gz_path[:-3] + ".txt"
        try:
            with gzip.open(gz_path, "rt", encoding="ascii", errors="replace") as f_in:
                with open(txt_path, "w", encoding="ascii") as f_out_txt:
                    f_out_txt.write(f_in.read())

            result = subprocess.run(
                ["condor_history", "-file", txt_path, "-af:ht"] + FIELDS,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            os.remove(txt_path)

            if result.stderr.strip():
                print(f"⚠️ condor_history error in {gz_path}:\n{result.stderr.strip()}")
                continue

            lines = result.stdout.strip().split("\n")
            for line in lines:
                values = line.split("\t")
                # 填补缺失字段数量
                while len(values) < len(FIELDS):
                    values.append("undefined")
                writer.writerow(values[:len(FIELDS)])

        except Exception as e:
            print(f"❌ Error processing {gz_path}: {e}")
