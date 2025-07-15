#!/bin/bash

set -e

RAW_BASE="/home/master/wzheng/projects/htcondor_eda/data"
OUT_BASE="/home/master/wzheng/projects/htcondor_eda/results/condor_cli_extracted"
mkdir -p "$OUT_BASE"

FIELDS="ClusterId ProcId Owner OwnerGroup Cmd SUBMIT_Cmd Arguments Args JobStatus ExitCode EnteredCurrentStatus CompletionDate JobStartDate RemoteWallClockTime CumulativeRemoteUserCpu CumulativeRemoteSysCpu CumulativeSuspensionTime ResidentSetSize_RAW ImageSize_RAW RequestCpus RequestMemory RequestDisk NumJobStarts JobRunCount LastRemoteHost Iwd SubmitHost GlobalJobId x509UserProxyVOName MATCH_GLIDEIN_Site MATCH_GLIDEIN_ResourceName MATCH_EXP_JOB_GLIDEIN_Entry_Name LastRemoteWallClockTime"

for NODE in $(ls "$RAW_BASE"); do
    NODE_DIR="$RAW_BASE/$NODE"
    OUTFILE="$OUT_BASE/parsed_${NODE}.csv"
    LOGFILE="$OUT_BASE/fallback_lines_${NODE}.log"
    echo "Processing $NODE_DIR $OUTFILE"

    echo "$FIELDS" | tr ' ' ',' > "$OUTFILE"
    > "$LOGFILE"

    for GZFILE in "$NODE_DIR"/history.*.gz; do
        TXTFILE="${GZFILE%.gz}.txt"
        gunzip -c "$GZFILE" > "$TXTFILE"

        condor_history -file "$TXTFILE" -af:ht $FIELDS | awk -v fallbacklog="$LOGFILE" -v out="$OUTFILE" -F'\t' '
        BEGIN {
            fallback_count = 0
            OFS = ","
        }
        tolower($1) == "clusterid" && tolower($2) == "procid" && tolower($3) == "owner" {
            print $0 >> fallbacklog
            fallback_count++
            next
        }
        {
            quoted_line = ""
            for (j = 1; j <= NF; j++) {
                gsub(/"/, """", $j)
                quoted_line = quoted_line "\"" $j "\"" (j < NF ? OFS : "")

            }
            print quoted_line >> out
        }
        END {
            print " Fallback lines skipped: " fallback_count > "/dev/stderr"
        }'

        rm -f "$TXTFILE"
    done
done

echo "All nodes processed."
