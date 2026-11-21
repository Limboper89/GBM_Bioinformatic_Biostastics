"""
Analysis Script: RSI Gene Expression and Kaplan-Meier Survival Analysis in TCGA-GBM Radiation-Treated Patients
Date: October 2026
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Set plot style
sns.set_theme(style="ticks", font_scale=1.05)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Directories
work_dir = r"d:\DATA\GBM_bioinormatics\GBM_analysis"
out_dir = os.path.join(work_dir, "RSI_radiation_survival")
plots_dir = os.path.join(out_dir, "plots")
tables_dir = os.path.join(out_dir, "tables")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# 1. Define Torres-Roca et al. RSI algorithm coefficients
rsi_coeffs = {
    "AR": -0.0098009,
    "JUN": 0.0128283,
    "STAT1": 0.0254552,
    "PRKCB": -0.0017589,
    "RELA": -0.0038171,
    "ABL1": 0.1070213,
    "SUMO1": -0.0002509,
    "PAK2": -0.0092431,
    "HDAC1": -0.0204469,
    "IRF1": -0.0441683
}
rsi_genes = list(rsi_coeffs.keys())
coeffs_series = pd.Series(rsi_coeffs)

# Gene functional annotations
gene_annotations = {
    "ABL1": ("c-ABL Tyrosine Kinase", "DNA Damage Response / Pro-survival", "Radioresistant"),
    "STAT1": ("Signal Transducer 1", "Interferon Signaling", "Radioresistant"),
    "JUN": ("c-Jun Transcription Factor", "AP-1 / Stress Response", "Radioresistant"),
    "SUMO1": ("Small Ubiquitin Modifier 1", "Post-translational Modification", "Radiosensitive"),
    "PRKCB": ("Protein Kinase C Beta", "Ser/Thr Kinase Signaling", "Radiosensitive"),
    "RELA": ("NF-kB p65 Subunit", "NF-kB Pro-survival Pathway", "Radiosensitive"),
    "PAK2": ("p21-Activated Kinase 2", "Cytoskeleton & Apoptosis", "Radiosensitive"),
    "AR": ("Androgen Receptor", "Nuclear Hormone Receptor", "Radiosensitive"),
    "HDAC1": ("Histone Deacetylase 1", "Epigenetic Chromatin Remodeling", "Radiosensitive"),
    "IRF1": ("Interferon Regulatory Factor 1", "Antitumor Immunity & Apoptosis", "Radiosensitive")
}

print("Loading TCGA clinical and expression datasets...")
master_path = os.path.join(work_dir, "TIGER_analysis", "treatment_segregated_analysis", "tcga_gbm_treatment_serine_master.csv")
fpkm_path = os.path.join(tables_dir, "tcga_gbm_rsi_expression_fpkm.csv")
tpm_path = os.path.join(tables_dir, "tcga_gbm_rsi_expression_tpm.csv")

df_master = pd.read_csv(master_path)
df_fpkm = pd.read_csv(fpkm_path)
df_tpm = pd.read_csv(tpm_path)

# Filter to primary tumors (-01)
df_fpkm_01 = df_fpkm[df_fpkm['Sample_ID'].str.endswith('-01')].copy()
df_fpkm_01['patient_barcode'] = df_fpkm_01['Sample_ID'].str.slice(0, 12)

df_tpm_01 = df_tpm[df_tpm['Sample_ID'].str.endswith('-01')].copy()
df_tpm_01['patient_barcode'] = df_tpm_01['Sample_ID'].str.slice(0, 12)

# Compute Rank-based RSI: ranks (1 to 10 within sample) multiplied by coefficients
ranks_fpkm = df_fpkm_01[rsi_genes].rank(axis=1)
df_fpkm_01['RSI_rank'] = (ranks_fpkm * coeffs_series).sum(axis=1)

# Compute Log2 FPKM RSI
log2_fpkm = np.log2(df_fpkm_01[rsi_genes] + 1)
df_fpkm_01['RSI_log2_fpkm'] = (log2_fpkm * coeffs_series).sum(axis=1)

# Compute Log2 TPM RSI
log2_tpm = np.log2(df_tpm_01[rsi_genes] + 1)
df_tpm_01['RSI_log2_tpm'] = (log2_tpm * coeffs_series).sum(axis=1)

# Merge datasets
df_merged = pd.merge(df_master, df_fpkm_01[['patient_barcode', 'Sample_ID', 'RSI_rank', 'RSI_log2_fpkm'] + rsi_genes + [f"{g}_log2" for g in rsi_genes]], on='patient_barcode')
df_merged = pd.merge(df_merged, df_tpm_01[['patient_barcode', 'RSI_log2_tpm'] + [g for g in rsi_genes]], on='patient_barcode', suffixes=('', '_tpm'))

# Rename RSI_rank to primary RSI
df_merged['RSI'] = df_merged['RSI_rank']

print(f"Merged TCGA cohort size: N = {len(df_merged)} patients")
print(f"Treatment breakdown:\n{df_merged['treatment_group'].value_counts()}")

# Export master merged dataset
excel_master_path = os.path.join(tables_dir, "TCGA_GBM_RSI_patient_master.xlsx")
csv_master_path = os.path.join(tables_dir, "TCGA_GBM_RSI_patient_master.csv")
df_merged.to_csv(csv_master_path, index=False)
with pd.ExcelWriter(excel_master_path, engine='openpyxl') as writer:
    df_merged.to_excel(writer, sheet_name="Master_Patient_Cohort", index=False)
    df_merged[df_merged['treatment_group'] == 'Radiation Only'].to_excel(writer, sheet_name="Radiation_Only_Cohort", index=False)
    df_merged[(df_merged['has_rad'] == True) & (df_merged['has_immuno'] == False)].to_excel(writer, sheet_name="All_Rad_Non_Immuno", index=False)
print("Saved master datasets to tables directory.")
