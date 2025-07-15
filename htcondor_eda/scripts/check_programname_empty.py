# -*- coding: utf-8 -*-

import pandas as pd
import os

df = pd.read_csv("/home/master/wzheng/projects/htcondor_eda/results/merged_all_nodes_clean.csv", dtype=str, keep_default_na=False)

def extract_prog_name(cmd):
    if pd.isnull(cmd) or cmd.strip() == '':
        return ''
    return os.path.basename(cmd.strip().split()[0])

df['ProgramName'] = df['Cmd'].apply(extract_prog_name)
empty_name_mask = df['ProgramName'] == ''
print(" ProgramName empty total:", empty_name_mask.sum())
print(df.loc[empty_name_mask, ['Cmd', 'Owner', 'GlobalJobId', 'source_node']].head(20))
