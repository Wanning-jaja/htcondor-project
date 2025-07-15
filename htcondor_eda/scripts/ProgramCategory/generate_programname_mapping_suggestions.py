
# -*- coding: utf-8 -*-
import pandas as pd
import re
from collections import defaultdict

# =========================
# v9.9 ProgramName 聚类建议脚本
# 特点：
# ✅ 保留所有人工规则（前缀 + 正则结构）
# ✅ 去除你明确指出的两条规则（Q2_1b_ 前缀 + Sys_mc16a_AF.sh 正则）
# ✅ 行级聚合统计，保证 MatchedNamesCount ≠ TotalJobCount
# ✅ fallback 自动结构抽象匹配
# =========================

# ==== 输入路径 ====
others_path = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory/ProgramName_others_only.csv"
lowfreq_path = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory/ProgramName_lowfreq_only.csv"

# ==== 加载原始记录 ====
df_raw = pd.concat([
    pd.read_csv(others_path),
    pd.read_csv(lowfreq_path)
], ignore_index=True)

df_raw['ProgramName'] = df_raw['ProgramName'].astype(str).str.strip()
df_raw = df_raw[df_raw['ProgramName'] != '']
records = df_raw.to_dict(orient='records')
# ==== 构建作业计数字典 ====
job_counter = df_raw['ProgramName'].value_counts().to_dict()
# ==== 提取唯一程序名用于聚类 ====
all_names = df_raw['ProgramName'].drop_duplicates().tolist()

# ==== 显式结构匹配规则（正则） ====
explicit_patterns = {
    r"^DIRAC_[\w\d_]+_pilotwrapper\.py$": "Dirac_pilotwrapper",
    r"^toy_\d+_l_VLL_M\d+.*$": "toy_l_VLL_M",
    r"^Sys\d+_mc\d+d_\d+_AF$": "Sys_mcd_AF",
    r"^Sys\d+_mc\d+a_\d+_AF\.root$": "Sys_mca_AF_root",
    r"^Pilot__c\d{2}m\d{2}__.*$": "Pilot_cNmN",
    r"^Pilot__mchuge__.*$": "Pilot_mchuge",
    r"^Pilot__mcxxl__.*$": "Pilot_mcxxl",
    r"^Pilot__mcsmall__.*$": "Pilot_mcsmall",
    r"^Pilot__mcmedium__.*$": "Pilot_mcmedium",
    r"^Pilot__sclong__.*$": "Pilot_sclong",
    r"^Pilot__schuge__.*$": "Pilot_schuge",
    r"^Pilot__schimem__.*$": "Pilot_schimem",
    r"^Pilot__scmedium__.*$": "Pilot_scmedium",
    r"^Q2_1b_\d+_\d+$": "Q2_1b_numeric",
    r"^1Z_0b_1SFOS_\d+_\d+$": "1Z_SFOS_numeric"
}

# ==== 显式前缀匹配规则（prefix） ====
explicit_prefixes = [
    "VR_1tau", "SR_1tau", "VR_1l1tauOS2b", "CR_1tau", "Fit_Asimov_",
    "regionVR_", "regionSR_", "regionCR_", "run_all_", "submit_star_",
    "run_compute_", "run_model_", "run_nplm_", "Fit_SplusB_",
    "VLL_M1000_", "VLL_M1100_", "VLL_M1200_", "VLL_M1300_", "VLL_M1400_",
    "histograms_Wjets_", "scripts_RANKINGEL_", "Q2_0b_", "0Z_0b_0SFOS_",
    "CombineFinal_Sys1_", "CombineFinal_Sys2", "SplusBAsimovStatOnly_1tauBJET",
    "VR_2tau", "SR_2tau", "CR_2tau", "VR_LnT1tau_", "RecombineNominal_",
    "user.jmharris.700", "BonlyDataUnblinded_1tauBJET_l_VLL_M",
    "executable_user.adsalvad."
]

# ==== 行级聚类处理 ====
pattern_to_rows = defaultdict(list)

for row in records:
    name = row['ProgramName']
    matched = False

    # 1. 明确结构规则匹配（最高优先级）
    for pat in explicit_patterns:
        if re.fullmatch(pat, name):
            pattern_to_rows[pat].append(name)
            matched = True
            break
    if matched:
        continue

    # 2. 明确前缀规则匹配
    for pfx in explicit_prefixes:
        if name.startswith(pfx):
            pattern_to_rows[f"^{re.escape(pfx)}.*$"].append(name)
            matched = True
            break
    if matched:
        continue

    # 3. fallback：结构抽象聚合
    name_abstract = re.sub(r"[a-fA-F0-9]{6,}", "<HASH>", name)
    name_abstract = re.sub(r"\d+", "<NUM>", name_abstract)
    name_abstract = re.sub(r"[_.-]+", "_", name_abstract)
    pattern_to_rows[f"^{re.escape(name_abstract)}$"].append(name)

# ==== 汇总输出 ====
rows = []
for pattern, names in pattern_to_rows.items():
    unique_names = sorted(set(names))
    rows.append({
        "RegexPattern": pattern,
        "MatchedNamesCount": len(unique_names),
        "TotalJobCount": len(names),
        "SampleProgramNames": "; ".join(unique_names[:5])
    })

out_df = pd.DataFrame(rows).sort_values("TotalJobCount", ascending=False)
out_path = "/home/master/wzheng/projects/htcondor_eda/results/ProgramCategory/program_name_mapping_suggestions_v9.csv"
out_df.to_csv(out_path, index=False)
print(f"saved to {out_path}")
