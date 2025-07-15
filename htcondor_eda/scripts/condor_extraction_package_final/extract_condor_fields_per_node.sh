#!/bin/bash

set -e

RAW_BASE="/home/master/wzheng/projects/htcondor_eda/data"
OUT_BASE="/home/master/wzheng/projects/htcondor_eda/results/condor_cli_extracted"
mkdir -p "$OUT_BASE"

FIELDS="ClusterId ProcId Owner OwnerGroup Cmd SUBMIT_Cmd Arguments Args JobStatus ExitCode EnteredCurrentStatus CompletionDate JobStartDate RemoteWallClockTime CumulativeRemoteUserCpu CumulativeRemoteSysCpu CumulativeSuspensionTime ResidentSetSize_RAW ImageSize_RAW RequestCpus RequestMemory RequestDisk NumJobStarts JobRunCount LastRemoteHost Iwd SubmitHost GlobalJobId x509UserProxyVOName MATCH_GLIDEIN_Site MATCH_GLIDEIN_ResourceName MATCH_EXP_JOB_GLIDEIN_Entry_Name LastRemoteWallClockTime"

for NODE in $(ls "$RAW_BASE"); do
    NODE_DIR="$RAW_BASE/$NODE"
    OUTFILE="$OUT_BASE/parsed_${NODE}.csv"
    echo "Processing $NODE_DIR → $OUTFILE"

    # 写入 CSV header
    echo "$FIELDS" | tr ' ' ',' > "$OUTFILE"

    for GZFILE in "$NODE_DIR"/history.*.gz; do
        TXTFILE="${GZFILE%.gz}.txt"
        gunzip -c "$GZFILE" > "$TXTFILE"

        # 过滤掉 fallback header 行
        condor_history -file "$TXTFILE" -af:ht $FIELDS | \
        awk -F'\t' 'tolower($1) != "clusterid" && tolower($2) != "procid" && tolower($3) != "owner"' >> "$OUTFILE"

        rm -f "$TXTFILE"
    done
done

echo "✅ All nodes processed."
