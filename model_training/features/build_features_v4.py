# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import argparse
import json
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Config & constants
# ---------------------------
DEFAULT_INPUT  = "/home/master/wzheng/projects/model_training/data/programID_grouped_cleaned(withOwnerGroup).csv"
DEFAULT_OUTPUT = "/home/master/wzheng/projects/model_training/data/model_features_v4.csv"
ENCODER_DIR    = "/home/master/wzheng/projects/model_training/data/encoders"
PID_MAP_DEFAULT = "/home/master/wzheng/projects/model_training/data/ProgramID_encoding_map.csv"

TARGET_COL = "RemoteWallClockTime"
TIME_COL   = "SubmitTime"

# ---------------------------
# Helpers
# ---------------------------
def to_datetime_series(epoch_s: pd.Series) -> pd.Series:
    # epoch seconds -> pandas datetime (UTC)
    return pd.to_datetime(epoch_s.astype(float), unit="s", utc=True, errors="coerce")

def extract_cluster(global_job_id: str) -> str:
    """
    GlobalJobId example: 'submit01.pic.es#8072098.0#1703494717'
    take the segment before first '#' then take prefix before first '.' -> 'submit01'
    """
    if not isinstance(global_job_id, str) or "#" not in global_job_id:
        return "unknown"
    host = global_job_id.split("#", 1)[0]
    if "." in host:
        return host.split(".", 1)[0]
    return host or "unknown"

def freq_encode(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Frequency encoding: return (count, freq) where freq = count / len(s)
    """
    counts = s.value_counts(dropna=False)
    count_map = counts.to_dict()
    total = float(len(s))
    freq_map = {k: v / total for k, v in count_map.items()}
    return s.map(count_map), s.map(freq_map)

def _bayes_smooth(mean, count, global_mean, alpha):
    return (mean * count + global_mean * alpha) / (count + alpha)

def time_based_oof_target_encoding(
    df: pd.DataFrame,
    key_col: str,
    target_col: str,
    time_col: str,
    n_splits: int = 5,
    alpha: float = 10.0,
    min_count: int = 5,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Time-forward OOF target mean encoding.
    - Sort by time; cut into n_splits by quantiles.
    - For each fold: fit on all PAST folds, encode CURRENT fold. Low-freq/new keys fallback to global mean (Bayesian smoothing).
    - Return: OOF-encoded series (aligned to the provided df) and the full-fit mapping (for val/test transform).
    """
    df = df.copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    times = df[time_col].values
    qs = np.linspace(0, 1, n_splits + 1)
    cut_points = np.quantile(times, qs)
    # avoid identical boundaries
    cut_points[0] -= 1e-3
    cut_points[-1] += 1e-3

    oof = np.zeros(len(df), dtype=float)
    global_mean = float(df[target_col].mean())

    for i in range(n_splits):
        left  = cut_points[0 if i == 0 else i]
        right = cut_points[i + 1]

        # current (future) slice
        val_idx = (times > left) & (times <= right)
        # train = past slices
        tr_idx  = times <= left

        if not tr_idx.any():
            # first slice has no past ¡ú fallback to global mean
            oof[val_idx] = global_mean
            continue

        tr_df = df.loc[tr_idx, [key_col, target_col]]
        agg = tr_df.groupby(key_col)[target_col].agg(["mean", "count"]).reset_index()

        # Bayesian smoothing
        agg["enc"] = _bayes_smooth(agg["mean"], agg["count"], global_mean, alpha=alpha)
        key_to_enc = dict(zip(agg[key_col].astype(str), agg["enc"].astype(float)))

        # encode current slice
        val_keys = df.loc[val_idx, key_col].astype(str)
        enc_vals = val_keys.map(key_to_enc)
        enc_vals = enc_vals.fillna(global_mean)
        oof[val_idx] = enc_vals.values

    # fit on all for val/test transform
    full_agg = df.groupby(key_col)[target_col].agg(["mean", "count"]).reset_index()
    full_agg["enc"] = _bayes_smooth(full_agg["mean"], full_agg["count"], global_mean, alpha=alpha)
    mapping = {str(k): float(v) for k, v in zip(full_agg[key_col].astype(str), full_agg["enc"].astype(float))}
    mapping["__global__"] = global_mean

    # keep the original (sorted) index alignment of this df
    return pd.Series(oof, index=df.index).sort_index(), mapping

def apply_target_mapping(keys: pd.Series, mapping: Dict[str, float]) -> pd.Series:
    """Apply the fitted mapping to new data (unknown keys fallback to global)."""
    global_mean = mapping.get("__global__", float("nan"))
    vals = keys.astype(str).map(mapping)
    return vals.fillna(global_mean)

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(ENCODER_DIR, exist_ok=True)

    # ====== Robust CSV read & numeric coercion ======
    NUMERIC_COLS = [
        "RequestCpus", "RequestMemory", "RequestDisk",
        "ResidentSetSize_RAW", "ImageSize_RAW",
        "NumJobStarts", "JobRunCount", "JobStatus", "ExitCode",
        "RemoteWallClockTime", "SubmitTime", "JobCount",
    ]

    df = pd.read_csv(
        args.input,
        dtype={
            "ResidentSetSize_RAW": "string",
            "ImageSize_RAW": "string",
        },
        na_values=["", " ", "NA", "N/A", "null", "NULL", "NaN", "nan", "Undefined", "undefined", "?", "-"],
        keep_default_na=True,
        low_memory=False,
    )

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Guards
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing time column {TIME_COL}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column {TARGET_COL}")

    # ========== Base derivations ==========
    dt = to_datetime_series(df[TIME_COL])
    df["DOW"] = dt.dt.dayofweek.astype("Int64")   # 0=Mon ... 6=Sun
    df["Hour"] = dt.dt.hour.astype("Int64")

    if "GlobalJobId" in df.columns:
        df["Cluster"] = df["GlobalJobId"].astype(str).map(extract_cluster)
    else:
        df["Cluster"] = "unknown"

    # Resource density (use RequestCpus_safe)
    df["RequestCpus_safe"] = df["RequestCpus"].replace(0, np.nan)
    df["ResourceDensity"]  = df["RequestMemory"] / (df["RequestCpus_safe"].astype(float) + 1e-6)
    df["LogResourceDensity"] = np.log1p(df["ResourceDensity"].clip(lower=0))

    # ========== Cross keys ==========
    crosses: List[Tuple[str, str, str]] = []
    if "ProgramPath4" in df.columns:
        crosses.append(("ProgramPath4", "DOW", "PP4__DOW"))
    if "OwnerGroup" in df.columns and "Cluster" in df.columns:
        crosses.append(("OwnerGroup", "Cluster", "OG__CLUST"))
    if "Queue" in df.columns:
        crosses.append(("OwnerGroup", "Queue", "OG__Q"))

    for a, b, name in crosses:
        df[name] = df[a].astype(str) + "|" + df[b].astype(str)

    # ========== Frequency encodings ==========
    freq_targets = []
    if "PP4__DOW" in df.columns:
        freq_targets.append("PP4__DOW")
    if "OG__CLUST" in df.columns:
        freq_targets.append("OG__CLUST")
    if "OG__Q" in df.columns:
        freq_targets.append("OG__Q")

    for col in freq_targets:
        cnt, freq = freq_encode(df[col].astype(str))
        df[f"{col}__count"] = cnt.astype("Int64")
        df[f"{col}__freq"]  = freq.astype(float)

    # ========== OOF target mean encodings (time-forward) ==========
    oof_targets = [c for c in ["PP4__DOW", "OG__CLUST", "OG__Q"] if c in df.columns]

    encoder_index: Dict[str, str] = {}
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    for col in oof_targets:
        oof_base = df[[col, TARGET_COL, TIME_COL]].dropna(subset=[col, TARGET_COL, TIME_COL])
        if oof_base.empty:
            continue

        oof_enc, mapping = time_based_oof_target_encoding(
            df=oof_base[[col, TARGET_COL, TIME_COL]].copy(),
            key_col=col,
            target_col=TARGET_COL,
            time_col=TIME_COL,
            n_splits=args.oof_splits,
            alpha=args.oof_alpha,
            min_count=args.oof_min_count,
        )

        new_col = f"{col}__te_mean_oof"
        full_series = pd.Series(np.nan, index=df.index, dtype=float)
        full_series.loc[oof_base.index] = oof_enc.values
        df[new_col] = full_series.values

        map_path = os.path.join(ENCODER_DIR, f"{col}__te_mean.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)
        encoder_index[new_col] = map_path

    with open(os.path.join(ENCODER_DIR, "_INDEX.json"), "w", encoding="utf-8") as f:
        json.dump(encoder_index, f, ensure_ascii=False, indent=2)

    # ========== ProgramID encoding merge ==========
    df["ProgramID"] = df["ProgramID"].astype(str)
    if os.path.exists(args.pid_map):
        map_df = pd.read_csv(args.pid_map)
        required_cols = {"ProgramID_str", "ProgramID_encoded"}
        if not required_cols.issubset(set(map_df.columns)):
            raise ValueError(f"PID map missing required columns: {required_cols}")
        before_rows = len(df)
        df = df.merge(map_df[["ProgramID_str", "ProgramID_encoded"]],
                      left_on="ProgramID", right_on="ProgramID_str", how="left")
        df.drop(columns=["ProgramID_str"], inplace=True)

        miss_mask = df["ProgramID_encoded"].isna()
        df.loc[miss_mask, "ProgramID_encoded"] = -1  # Others
        df["ProgramID_encoded"] = df["ProgramID_encoded"].astype("int64")

        mapped = (~miss_mask).sum()
        print(f"[OK] ProgramID encoding merged from: {args.pid_map}")
        print(f"Rows total: {before_rows} | mapped: {mapped} | set to -1 (Others): {miss_mask.sum()}")
        if miss_mask.any():
            sample_unmapped = df.loc[miss_mask, "ProgramID"].astype(str).dropna().unique()[:5]
            print("Sample of unmapped ProgramID:", list(sample_unmapped))
    else:
        print(f"[WARN] ProgramID map not found: {args.pid_map} -> skip encoding merge. "
              f"Downstream scripts may require ProgramID_encoded.")

    # ========== Output ==========
    df.to_csv(args.output, index=False)
    print(f"[OK] Write down the characteristics: {args.output}")
    print(f"[OK] The encoder is stored in: {ENCODER_DIR}")
    print("New column example:", [c for c in df.columns if c.endswith("__count") or c.endswith("__freq") or c.endswith("__te_mean_oof")][:10])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v5.2 Feature engineering (cross-features + time-based OOF target encoding)")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV (including new features)")
    parser.add_argument("--oof_splits", type=int, default=5, help="OOF Forward fold count")
    parser.add_argument("--oof_alpha", type=float, default=10.0, help="OOF Bayesian smoothing alpha")
    parser.add_argument("--oof_min_count", type=int, default=5, help="Low frequency threshold (not used directly, reserved)")
    parser.add_argument("--pid_map", type=str, default=PID_MAP_DEFAULT, help="CSV mapping: ProgramID_str -> ProgramID_encoded")
    args = parser.parse_args()
    main(args)
