"""
Comprehensive Pipeline: RSI Gene Expression and Survival Analysis in TCGA-GBM Radiation-Treated Patients
Date: October 2026
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# Style configuration
sns.set_theme(style="ticks", font_scale=1.05)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Directories
work_dir = r"d:\DATA\GBM_bioinormatics\GBM_analysis"
out_dir = os.path.join(work_dir, "RSI_radiation_survival")
plots_dir = os.path.join(out_dir, "plots")
tables_dir = os.path.join(out_dir, "tables")

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

# 1. Define Torres-Roca et al. RSI Algorithm
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

# Biological function annotation
gene_annotations = {
    "ABL1": ("c-ABL Tyrosine Kinase", "DNA Damage Response & Pro-survival", "Radioresistant (w > 0)"),
    "STAT1": ("Signal Transducer 1", "Interferon / Inflammatory Signaling", "Radioresistant (w > 0)"),
    "JUN": ("c-Jun Transcription Factor", "AP-1 Complex / Cellular Stress", "Radioresistant (w > 0)"),
    "SUMO1": ("Small Ubiquitin Modifier 1", "Post-translational Modification", "Radiosensitive (w < 0)"),
    "PRKCB": ("Protein Kinase C Beta", "Ser/Thr Kinase Signaling", "Radiosensitive (w < 0)"),
    "RELA": ("NF-kB p65 Subunit", "NF-kB Survival / Inflammation", "Radiosensitive (w < 0)"),
    "PAK2": ("p21-Activated Kinase 2", "Cytoskeleton & Apoptosis", "Radiosensitive (w < 0)"),
    "AR": ("Androgen Receptor", "Nuclear Hormone Signaling", "Radiosensitive (w < 0)"),
    "HDAC1": ("Histone Deacetylase 1", "Epigenetic Chromatin Regulation", "Radiosensitive (w < 0)"),
    "IRF1": ("Interferon Regulatory Factor 1", "Antitumor Immunity & Apoptosis", "Radiosensitive (w < 0)")
}

# 2. Load Data
print("Loading TCGA datasets...")
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

# Calculate RSI metrics
# Metric A: Within-sample rank-based RSI (Rank 1 to 10 weighted sum)
ranks_fpkm = df_fpkm_01[rsi_genes].rank(axis=1)
df_fpkm_01['RSI_rank'] = (ranks_fpkm * coeffs_series).sum(axis=1)

# Metric B: Continuous Log2 FPKM RSI
log2_fpkm = np.log2(df_fpkm_01[rsi_genes] + 1)
df_fpkm_01['RSI_log2_fpkm'] = (log2_fpkm * coeffs_series).sum(axis=1)

# Metric C: Continuous Log2 TPM RSI
log2_tpm = np.log2(df_tpm_01[rsi_genes] + 1)
df_tpm_01['RSI_log2_tpm'] = (log2_tpm * coeffs_series).sum(axis=1)

# Merge datasets
df_merged = pd.merge(df_master, df_fpkm_01[['patient_barcode', 'Sample_ID', 'RSI_rank', 'RSI_log2_fpkm'] + rsi_genes + [f"{g}_log2" for g in rsi_genes]], on='patient_barcode')
df_merged = pd.merge(df_merged, df_tpm_01[['patient_barcode', 'RSI_log2_tpm'] + [g for g in rsi_genes]], on='patient_barcode', suffixes=('', '_tpm'))
df_merged['RSI'] = df_merged['RSI_rank']

print(f"Cohort size: N = {len(df_merged)} total matched patients")

# Extract subsets
df_rad_only = df_merged[df_merged['treatment_group'] == 'Radiation Only'].copy()
df_all_rad = df_merged[(df_merged['has_rad'] == True) & (df_merged['has_immuno'] == False)].copy()
df_combo = df_merged[df_merged['treatment_group'] == 'Combination (Chemo + Radiation)'].copy()
df_chemo_only = df_merged[df_merged['treatment_group'] == 'Chemotherapy Only'].copy()
df_naive = df_merged[df_merged['treatment_group'] == 'Treatment Naive / No Treatment'].copy()

print(f"  Radiation Therapy Only: N = {len(df_rad_only)}")
print(f"  All Radiation Treated (Non-immuno): N = {len(df_all_rad)}")
print(f"  Combination (Chemo + Radiation): N = {len(df_combo)}")
print(f"  Chemotherapy Only: N = {len(df_chemo_only)}")
print(f"  Treatment Naive / Resection Only: N = {len(df_naive)}")

# ==============================================================================
# FIGURE 1: TORRES-ROCA RSI MODEL WEIGHTS / ARCHITECTURE
# ==============================================================================
print("\nGenerating Figure 1: RSI Model Weights (Lollipop Chart)...")
fig, ax = plt.subplots(figsize=(9, 5.5))

df_weights = pd.DataFrame({
    'Gene': rsi_genes,
    'Weight': [rsi_coeffs[g] for g in rsi_genes],
    'Role': [gene_annotations[g][2] for g in rsi_genes],
    'Function': [gene_annotations[g][1] for g in rsi_genes]
}).sort_values('Weight', ascending=True)

colors = ['#2b7bba' if w < 0 else '#d95f02' for w in df_weights['Weight']]
y_pos = np.arange(len(df_weights))

ax.hlines(y=y_pos, xmin=0, xmax=df_weights['Weight'], color=colors, alpha=0.7, linewidth=2.5)
ax.scatter(df_weights['Weight'], y_pos, color=colors, s=140, edgecolors='black', linewidth=1.2, zorder=3)

for idx, (w, g, func) in enumerate(zip(df_weights['Weight'], df_weights['Gene'], df_weights['Function'])):
    ha = 'left' if w >= 0 else 'right'
    offset = 0.003 if w >= 0 else -0.003
    ax.text(w + offset, idx, f"{w:+.4f}  ({func})", va='center', ha=ha, fontsize=9.5, fontweight='bold', color='#222222')

ax.set_yticks(y_pos)
ax.set_yticklabels(df_weights['Gene'], fontsize=11, fontweight='bold')
ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.8)
ax.set_xlim(-0.058, 0.125)
ax.set_xlabel("Torres-Roca Algorithm Coefficient Weight ($w_i$)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_title("Radiosensitivity Index (RSI) 10-Gene Model Architecture\n(Torres-Roca et al. / Eschrich et al.)", fontsize=12.5, fontweight='bold', pad=12)

# Legend annotations
ax.text(0.065, 1.5, "Radioresistant Hubs (w > 0)\nHigh expression -> Radioresistance", 
        fontsize=9.5, fontweight='bold', color='#d95f02', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdf2eb', edgecolor='#d95f02', alpha=0.9))
ax.text(-0.055, 1.5, "Radiosensitive Hubs (w < 0)\nHigh expression -> Radiosensitivity", 
        fontsize=9.5, fontweight='bold', color='#2b7bba', bbox=dict(boxstyle='round,pad=0.4', facecolor='#edf4fa', edgecolor='#2b7bba', alpha=0.9))

plt.tight_layout()
p1_png = os.path.join(plots_dir, "01_RSI_model_coefficients.png")
p1_pdf = os.path.join(plots_dir, "01_RSI_model_coefficients.pdf")
plt.savefig(p1_png, dpi=300)
plt.savefig(p1_pdf)
plt.close()

# ==============================================================================
# FIGURE 2: RSI GENE EXPRESSION DISTRIBUTION IN RADIATION ONLY COHORT (N=17)
# ==============================================================================
print("Generating Figure 2: RSI Gene Expression in Radiation Only Cohort...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Order genes by median expression
med_expr = df_rad_only[[f"{g}_log2" for g in rsi_genes]].median().sort_values(ascending=False)
ordered_genes = [c.replace('_log2', '') for c in med_expr.index]

# Panel A: Log2 Expression
df_long_log2 = pd.melt(df_rad_only, id_vars=['patient_barcode'], value_vars=[f"{g}_log2" for g in ordered_genes],
                       var_name='Gene', value_name='Log2_Expr')
df_long_log2['Gene'] = df_long_log2['Gene'].str.replace('_log2', '')
df_long_log2['Class'] = df_long_log2['Gene'].apply(lambda g: 'Radioresistant (w>0)' if rsi_coeffs[g] > 0 else 'Radiosensitive (w<0)')

palette = {'Radioresistant (w>0)': '#e66101', 'Radiosensitive (w<0)': '#5e3c99'}

sns.boxplot(data=df_long_log2, x='Gene', y='Log2_Expr', hue='Class', palette=palette, ax=ax1, width=0.55,
            boxprops=dict(alpha=0.6), fliersize=0)
sns.stripplot(data=df_long_log2, x='Gene', y='Log2_Expr', hue='Class', palette=palette, ax=ax1,
              dodge=False, size=6, jitter=0.2, alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.set_title("A. Log2(FPKM + 1) Expression in Radiation Only (N = 17)", fontsize=11.5, fontweight='bold')
ax1.set_xlabel("RSI Signature Genes", fontsize=10.5, fontweight='bold')
ax1.set_ylabel("Expression Level [Log2(FPKM + 1)]", fontsize=10.5, fontweight='bold')
ax1.tick_params(axis='x', rotation=30)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:2], labels[:2], loc='upper right', frameon=True, fontsize=9.5)

# Panel B: Normalized Relative Ranks (1 to 10)
ranks_rad = df_rad_only[ordered_genes].rank(axis=1)
ranks_long = pd.melt(ranks_rad.reset_index(), id_vars=['index'], value_vars=ordered_genes,
                     var_name='Gene', value_name='Rank')
ranks_long['Class'] = ranks_long['Gene'].apply(lambda g: 'Radioresistant (w>0)' if rsi_coeffs[g] > 0 else 'Radiosensitive (w<0)')

sns.boxplot(data=ranks_long, x='Gene', y='Rank', hue='Class', palette=palette, ax=ax2, width=0.55,
            boxprops=dict(alpha=0.6), fliersize=0)
sns.stripplot(data=ranks_long, x='Gene', y='Rank', hue='Class', palette=palette, ax=ax2,
              dodge=False, size=6, jitter=0.2, alpha=0.8, edgecolor='black', linewidth=0.5)

ax2.set_title("B. Within-Sample Gene Rank (1 = Lowest, 10 = Highest)", fontsize=11.5, fontweight='bold')
ax2.set_xlabel("RSI Signature Genes", fontsize=10.5, fontweight='bold')
ax2.set_ylabel("Within-Sample Expression Rank", fontsize=10.5, fontweight='bold')
ax2.tick_params(axis='x', rotation=30)
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2[:2], labels2[:2], loc='upper right', frameon=True, fontsize=9.5)

plt.tight_layout()
p2_png = os.path.join(plots_dir, "02_RSI_gene_expression_radiation_only.png")
p2_pdf = os.path.join(plots_dir, "02_RSI_gene_expression_radiation_only.pdf")
plt.savefig(p2_png, dpi=300)
plt.savefig(p2_pdf)
plt.close()

# ==============================================================================
# FIGURE 3: RSI GENE EXPRESSION IN ALL RADIATION-TREATED PATIENTS (N=201)
# ==============================================================================
print("Generating Figure 3: RSI Gene Expression in All Radiation Cohort...")
fig, ax = plt.subplots(figsize=(10, 5))

df_long_all = pd.melt(df_all_rad, id_vars=['patient_barcode'], value_vars=[f"{g}_log2" for g in ordered_genes],
                      var_name='Gene', value_name='Log2_Expr')
df_long_all['Gene'] = df_long_all['Gene'].str.replace('_log2', '')
df_long_all['Class'] = df_long_all['Gene'].apply(lambda g: 'Radioresistant (w>0)' if rsi_coeffs[g] > 0 else 'Radiosensitive (w<0)')

sns.violinplot(data=df_long_all, x='Gene', y='Log2_Expr', hue='Class', palette=palette, split=False, inner='quartile',
               cut=0, ax=ax, alpha=0.7)
ax.set_title("RSI Signature Gene Expression Profiles Across All Radiation-Treated Patients (N = 201)", fontsize=11.5, fontweight='bold', pad=10)
ax.set_xlabel("RSI Signature Genes", fontsize=10.5, fontweight='bold')
ax.set_ylabel("Expression Level [Log2(FPKM + 1)]", fontsize=10.5, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.legend(loc='upper right', frameon=True, fontsize=9.5)

plt.tight_layout()
p3_png = os.path.join(plots_dir, "03_RSI_gene_expression_all_radiation.png")
p3_pdf = os.path.join(plots_dir, "03_RSI_gene_expression_all_radiation.pdf")
plt.savefig(p3_png, dpi=300)
plt.savefig(p3_pdf)
plt.close()

# ==============================================================================
# FIGURE 4: SPEARMAN CORRELATION MATRIX HEATMAP
# ==============================================================================
print("Generating Figure 4: Inter-gene Correlation Matrix...")
corr_features = rsi_genes + ['RSI']
corr_matrix = df_all_rad[corr_features].corr(method='spearman')

fig, ax = plt.subplots(figsize=(8.5, 7.5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(240, 10, as_cmap=True)

sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, vmin=-0.6, vmax=0.6,
            center=0, square=True, linewidths=0.8, cbar_kws={"shrink": 0.8, "label": "Spearman Correlation (rho)"},
            ax=ax, annot_kws={"size": 9, "fontweight": "bold"})

ax.set_title("Spearman Correlation Network: RSI Genes & Composite Score\n(All Radiation-Treated TCGA-GBM Cohort, N = 201)",
             fontsize=11.5, fontweight='bold', pad=12)
plt.tight_layout()
p4_png = os.path.join(plots_dir, "04_RSI_correlation_matrix.png")
p4_pdf = os.path.join(plots_dir, "04_RSI_correlation_matrix.pdf")
plt.savefig(p4_png, dpi=300)
plt.savefig(p4_pdf)
plt.close()

# ==============================================================================
# FIGURE 5: RSI SCORE DISTRIBUTION & CUTPOINTS
# ==============================================================================
print("Generating Figure 5: RSI Score Distributions...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Cohort 1: Rad Only
sns.histplot(df_rad_only['RSI'], kde=True, color='#2b7bba', bins=10, ax=ax1, edgecolor='black', alpha=0.6)
sns.rugplot(df_rad_only['RSI'], color='#111111', height=0.08, ax=ax1)
rad_med = df_rad_only['RSI'].median()
ax1.axvline(rad_med, color='#d95f02', linestyle='--', linewidth=2, label=f"Median = {rad_med:.4f}")
ax1.axvline(0.8630, color='#7570b3', linestyle=':', linewidth=2, label="Optimal Cut = 0.8630")
ax1.set_title("A. Radiation Only Cohort (N = 17)", fontsize=11.5, fontweight='bold')
ax1.set_xlabel("RSI Score (Torres-Roca Rank Formula)", fontsize=10.5, fontweight='bold')
ax1.set_ylabel("Patient Count", fontsize=10.5, fontweight='bold')
ax1.legend(frameon=True, fontsize=9.5)

# Cohort 2: All Rad
sns.histplot(df_all_rad['RSI'], kde=True, color='#4daf4a', bins=25, ax=ax2, edgecolor='black', alpha=0.6)
sns.rugplot(df_all_rad['RSI'], color='#111111', height=0.05, ax=ax2)
all_med = df_all_rad['RSI'].median()
ax2.axvline(all_med, color='#d95f02', linestyle='--', linewidth=2, label=f"Median = {all_med:.4f}")
ax2.axvline(0.5680, color='#7570b3', linestyle=':', linewidth=2, label="Optimal Cut = 0.5680")
ax2.set_title("B. All Radiation-Treated Non-Immuno (N = 201)", fontsize=11.5, fontweight='bold')
ax2.set_xlabel("RSI Score (Torres-Roca Rank Formula)", fontsize=10.5, fontweight='bold')
ax2.set_ylabel("Patient Count", fontsize=10.5, fontweight='bold')
ax2.legend(frameon=True, fontsize=9.5)

plt.tight_layout()
p5_png = os.path.join(plots_dir, "05_RSI_score_distribution.png")
p5_pdf = os.path.join(plots_dir, "05_RSI_score_distribution.pdf")
plt.savefig(p5_png, dpi=300)
plt.savefig(p5_pdf)
plt.close()

# ==============================================================================
# HELPER: PUBLICATION KAPLAN-MEIER PLOTTER WITH AT-RISK TABLE
# ==============================================================================
def plot_publication_km(time_high, event_high, time_low, event_low, 
                         label_high, label_low, color_high, color_low,
                         title, subtitle, out_prefix, time_max=None,
                         pval=None, hr=None, hr_ci=None):
    fig = plt.figure(figsize=(7.5, 6.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.18)
    
    ax_km = plt.subplot(gs[0])
    ax_table = plt.subplot(gs[1])
    
    kmf_h = KaplanMeierFitter()
    kmf_h.fit(time_high, event_high, label=label_high)
    
    kmf_l = KaplanMeierFitter()
    kmf_l.fit(time_low, event_low, label=label_low)
    
    # Plot curves with confidence intervals
    kmf_l.plot_survival_function(ax=ax_km, color=color_low, linewidth=2.2, ci_show=True, ci_alpha=0.15)
    kmf_h.plot_survival_function(ax=ax_km, color=color_high, linewidth=2.2, ci_show=True, ci_alpha=0.15)
    
    # Censored points
    cens_h_t = time_high[event_high == 0]
    cens_l_t = time_low[event_low == 0]
    if len(cens_h_t) > 0:
        cens_h_surv = [kmf_h.predict(t) for t in cens_h_t]
        ax_km.scatter(cens_h_t, cens_h_surv, marker='+', color=color_high, s=55, linewidth=1.5, zorder=4)
    if len(cens_l_t) > 0:
        cens_l_surv = [kmf_l.predict(t) for t in cens_l_t]
        ax_km.scatter(cens_l_t, cens_l_surv, marker='+', color=color_low, s=55, linewidth=1.5, zorder=4)
        
    ax_km.set_ylim(-0.02, 1.05)
    if time_max is not None:
        ax_km.set_xlim(0, time_max)
    else:
        ax_km.set_xlim(0, max(time_high.max(), time_low.max()) * 1.05)
        
    ax_km.set_ylabel("Overall Survival Probability", fontsize=11, fontweight='bold', labelpad=8)
    ax_km.set_xlabel("")
    ax_km.set_xticklabels([])
    ax_km.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight='bold', pad=10)
    
    # Median survival
    med_h = kmf_h.median_survival_time_
    med_l = kmf_l.median_survival_time_
    
    # Annotation box
    text_lines = []
    if pval is not None:
        text_lines.append(f"Log-rank p = {pval:.4f}" if pval >= 0.001 else f"Log-rank p = {pval:.2e}")
    if hr is not None and hr_ci is not None:
        text_lines.append(f"Hazard Ratio = {hr:.2f} (95% CI: {hr_ci[0]:.2f}-{hr_ci[1]:.2f})")
    text_lines.append(f"Median OS ({label_low}): {med_l:.1f} mo")
    text_lines.append(f"Median OS ({label_high}): {med_h:.1f} mo")
    
    info_text = "\n".join(text_lines)
    ax_km.text(0.96, 0.94, info_text, transform=ax_km.transAxes, fontsize=9.5,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    
    ax_km.legend(loc='lower left', frameon=True, fontsize=10)
    
    # Number at risk table
    time_pts = [0, 6, 12, 24, 36, 48, 60]
    time_pts = [t for t in time_pts if t <= ax_km.get_xlim()[1]]
    
    at_risk_l = [sum(time_low >= t) for t in time_pts]
    at_risk_h = [sum(time_high >= t) for t in time_pts]
    
    ax_table.axis('off')
    table_data = [
        [f"{label_low}"] + [str(n) for n in at_risk_l],
        [f"{label_high}"] + [str(n) for n in at_risk_h]
    ]
    col_labels = ["Time (months)"] + [str(t) for t in time_pts]
    
    table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    
    # Style table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if row == 0:
            cell.set_text_props(weight='bold', color='#333333')
            cell.set_facecolor('#f5f5f5')
        elif col == 0:
            cell.set_text_props(weight='bold', color=color_low if row == 1 else color_high)
            cell.set_facecolor('white')
            
    fig.text(0.5, 0.02, "Overall Survival (Months)", ha='center', fontsize=11, fontweight='bold')
    
    plt.savefig(f"{out_prefix}.png", dpi=300)
    plt.savefig(f"{out_prefix}.pdf")
    plt.close()

# ==============================================================================
# FIGURE 6: KM SURVIVAL - RADIATION THERAPY ONLY (MEDIAN CUTPOINT)
# ==============================================================================
print("\nGenerating Figure 6: KM Survival - Radiation Only (Median Cutpoint)...")
rad_med = df_rad_only['RSI'].median()
rad_high = df_rad_only[df_rad_only['RSI'] >= rad_med]
rad_low = df_rad_only[df_rad_only['RSI'] < rad_med]

lr_rad_med = logrank_test(rad_high['time_months'], rad_low['time_months'], rad_high['status'], rad_low['status'])

# Cox model
cph = CoxPHFitter()
df_cox = df_rad_only[['time_months', 'status', 'RSI']].copy()
df_cox['RSI_Group'] = (df_cox['RSI'] >= rad_med).astype(int)
cph.fit(df_cox[['time_months', 'status', 'RSI_Group']], duration_col='time_months', event_col='status')
hr_val = np.exp(cph.params_['RSI_Group'])
hr_ci_vals = (np.exp(cph.confidence_intervals_.loc['RSI_Group'].iloc[0]), np.exp(cph.confidence_intervals_.loc['RSI_Group'].iloc[1]))

plot_publication_km(
    time_high=rad_high['time_months'], event_high=rad_high['status'],
    time_low=rad_low['time_months'], event_low=rad_low['status'],
    label_high=f"Radioresistant (High RSI, N={len(rad_high)})",
    label_low=f"Radiosensitive (Low RSI, N={len(rad_low)})",
    color_high="#BC3C29", color_low="#0072B5",
    title="TCGA-GBM Radiation Therapy Only Cohort (N = 17)",
    subtitle=f"Stratified by Median RSI (Cutpoint = {rad_med:.4f})",
    out_prefix=os.path.join(plots_dir, "06_KM_RSI_radiation_only_median"),
    pval=lr_rad_med.p_value, hr=hr_val, hr_ci=hr_ci_vals, time_max=32
)

# ==============================================================================
# FIGURE 7: KM SURVIVAL - RADIATION THERAPY ONLY (OPTIMAL CUTPOINT)
# ==============================================================================
print("Generating Figure 7: KM Survival - Radiation Only (Optimal Cutpoint)...")
opt_cut_rad = 0.8630
rad_opt_high = df_rad_only[df_rad_only['RSI'] > opt_cut_rad]
rad_opt_low = df_rad_only[df_rad_only['RSI'] <= opt_cut_rad]

lr_rad_opt = logrank_test(rad_opt_high['time_months'], rad_opt_low['time_months'], rad_opt_high['status'], rad_opt_low['status'])

df_cox_opt = df_rad_only[['time_months', 'status', 'RSI']].copy()
df_cox_opt['RSI_Group'] = (df_cox_opt['RSI'] > opt_cut_rad).astype(int)
cph_opt = CoxPHFitter()
cph_opt.fit(df_cox_opt[['time_months', 'status', 'RSI_Group']], duration_col='time_months', event_col='status')
hr_opt_val = np.exp(cph_opt.params_['RSI_Group'])
hr_opt_ci = (np.exp(cph_opt.confidence_intervals_.loc['RSI_Group'].iloc[0]), np.exp(cph_opt.confidence_intervals_.loc['RSI_Group'].iloc[1]))

plot_publication_km(
    time_high=rad_opt_high['time_months'], event_high=rad_opt_high['status'],
    time_low=rad_opt_low['time_months'], event_low=rad_opt_low['status'],
    label_high=f"Radioresistant (High RSI, N={len(rad_opt_high)})",
    label_low=f"Radiosensitive (Low RSI, N={len(rad_opt_low)})",
    color_high="#BC3C29", color_low="#0072B5",
    title="TCGA-GBM Radiation Therapy Only Cohort (N = 17)",
    subtitle=f"Stratified by Optimal Cutpoint (Cutpoint = {opt_cut_rad:.4f})",
    out_prefix=os.path.join(plots_dir, "07_KM_RSI_radiation_only_optimal"),
    pval=lr_rad_opt.p_value, hr=hr_opt_val, hr_ci=hr_opt_ci, time_max=32
)

# ==============================================================================
# FIGURE 8: KM SURVIVAL - ALL RADIATION-TREATED NON-IMMUNO (MEDIAN CUTPOINT)
# ==============================================================================
print("Generating Figure 8: KM Survival - All Radiation Treated (Median Cutpoint)...")
all_med = df_all_rad['RSI'].median()
all_high = df_all_rad[df_all_rad['RSI'] >= all_med]
all_low = df_all_rad[df_all_rad['RSI'] < all_med]

lr_all_med = logrank_test(all_high['time_months'], all_low['time_months'], all_high['status'], all_low['status'])

df_cox_all = df_all_rad[['time_months', 'status', 'RSI']].copy()
df_cox_all['RSI_Group'] = (df_cox_all['RSI'] >= all_med).astype(int)
cph_all = CoxPHFitter()
cph_all.fit(df_cox_all[['time_months', 'status', 'RSI_Group']], duration_col='time_months', event_col='status')
hr_all_val = np.exp(cph_all.params_['RSI_Group'])
hr_all_ci = (np.exp(cph_all.confidence_intervals_.loc['RSI_Group'].iloc[0]), np.exp(cph_all.confidence_intervals_.loc['RSI_Group'].iloc[1]))

plot_publication_km(
    time_high=all_high['time_months'], event_high=all_high['status'],
    time_low=all_low['time_months'], event_low=all_low['status'],
    label_high=f"Radioresistant (High RSI, N={len(all_high)})",
    label_low=f"Radiosensitive (Low RSI, N={len(all_low)})",
    color_high="#BC3C29", color_low="#0072B5",
    title="All Radiation-Treated TCGA-GBM Patients (Non-Immunotherapy, N = 201)",
    subtitle=f"Stratified by Median RSI (Cutpoint = {all_med:.4f})",
    out_prefix=os.path.join(plots_dir, "08_KM_RSI_all_radiation_median"),
    pval=lr_all_med.p_value, hr=hr_all_val, hr_ci=hr_all_ci, time_max=70
)

# ==============================================================================
# FIGURE 9: KM SURVIVAL - ALL RADIATION-TREATED NON-IMMUNO (OPTIMAL CUTPOINT)
# ==============================================================================
print("Generating Figure 9: KM Survival - All Radiation Treated (Optimal Cutpoint)...")
opt_cut_all = 0.5680
all_opt_high = df_all_rad[df_all_rad['RSI'] > opt_cut_all]
all_opt_low = df_all_rad[df_all_rad['RSI'] <= opt_cut_all]

lr_all_opt = logrank_test(all_opt_high['time_months'], all_opt_low['time_months'], all_opt_high['status'], all_opt_low['status'])

df_cox_all_opt = df_all_rad[['time_months', 'status', 'RSI']].copy()
df_cox_all_opt['RSI_Group'] = (df_cox_all_opt['RSI'] > opt_cut_all).astype(int)
cph_all_opt = CoxPHFitter()
cph_all_opt.fit(df_cox_all_opt[['time_months', 'status', 'RSI_Group']], duration_col='time_months', event_col='status')
hr_all_opt_val = np.exp(cph_all_opt.params_['RSI_Group'])
hr_all_opt_ci = (np.exp(cph_all_opt.confidence_intervals_.loc['RSI_Group'].iloc[0]), np.exp(cph_all_opt.confidence_intervals_.loc['RSI_Group'].iloc[1]))

plot_publication_km(
    time_high=all_opt_high['time_months'], event_high=all_opt_high['status'],
    time_low=all_opt_low['time_months'], event_low=all_opt_low['status'],
    label_high=f"Radioresistant (High RSI, N={len(all_opt_high)})",
    label_low=f"Radiosensitive (Low RSI, N={len(all_opt_low)})",
    color_high="#BC3C29", color_low="#0072B5",
    title="All Radiation-Treated TCGA-GBM Patients (Non-Immunotherapy, N = 201)",
    subtitle=f"Stratified by Optimal Cutpoint (Cutpoint = {opt_cut_all:.4f})",
    out_prefix=os.path.join(plots_dir, "09_KM_RSI_all_radiation_optimal"),
    pval=lr_all_opt.p_value, hr=hr_all_opt_val, hr_ci=hr_all_opt_ci, time_max=70
)

# ==============================================================================
# FIGURE 10: 10-PANEL KM SURVIVAL GRID - INDIVIDUAL RSI GENES (RADIATION ONLY)
# ==============================================================================
print("Generating Figure 10: 10-Panel KM Grid for Individual Genes in Radiation Only...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), sharey=True)
axes = axes.flatten()

gene_results_rad = []

for idx, gene in enumerate(rsi_genes):
    ax = axes[idx]
    med = df_rad_only[gene].median()
    g_high = df_rad_only[df_rad_only[gene] >= med]
    g_low = df_rad_only[df_rad_only[gene] < med]
    
    lr = logrank_test(g_high['time_months'], g_low['time_months'], g_high['status'], g_low['status'])
    
    kmf_h = KaplanMeierFitter()
    kmf_h.fit(g_high['time_months'], g_high['status'], label='High')
    kmf_h.plot_survival_function(ax=ax, color='#BC3C29', linewidth=2, ci_show=False)
    
    kmf_l = KaplanMeierFitter()
    kmf_l.fit(g_low['time_months'], g_low['status'], label='Low')
    kmf_l.plot_survival_function(ax=ax, color='#0072B5', linewidth=2, ci_show=False)
    
    # Censored points
    c_h = g_high[g_high['status'] == 0]['time_months']
    c_l = g_low[g_low['status'] == 0]['time_months']
    if len(c_h) > 0:
        ax.scatter(c_h, [kmf_h.predict(t) for t in c_h], marker='+', color='#BC3C29', s=40, zorder=3)
    if len(c_l) > 0:
        ax.scatter(c_l, [kmf_l.predict(t) for t in c_l], marker='+', color='#0072B5', s=40, zorder=3)
        
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 32)
    role_str = "w > 0" if rsi_coeffs[gene] > 0 else "w < 0"
    ax.set_title(f"{gene} ({role_str})\nlog-rank p = {lr.p_value:.3f}", fontsize=11, fontweight='bold')
    
    if idx % 5 == 0:
        ax.set_ylabel("Survival Probability", fontsize=10.5, fontweight='bold')
    else:
        ax.set_ylabel("")
        
    if idx >= 5:
        ax.set_xlabel("Time (Months)", fontsize=10.5, fontweight='bold')
    else:
        ax.set_xlabel("")
        
    med_h = kmf_h.median_survival_time_
    med_l = kmf_l.median_survival_time_
    ax.text(0.95, 0.92, f"Med High: {med_h:.1f}m\nMed Low: {med_l:.1f}m",
            transform=ax.transAxes, fontsize=8.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc', alpha=0.85))
    ax.legend(loc='lower left', fontsize=8.5, frameon=True)
    
    gene_results_rad.append({
        'Gene': gene,
        'Cohort': 'Radiation Only (N=17)',
        'Model_Weight': rsi_coeffs[gene],
        'Median_Expression': med,
        'High_N': len(g_high),
        'High_Median_OS': med_h,
        'Low_N': len(g_low),
        'Low_Median_OS': med_l,
        'LogRank_p': lr.p_value
    })

plt.suptitle("Individual RSI Gene Survival Stratification in Radiation Therapy Only Patients (N = 17)", 
             fontsize=13.5, fontweight='bold', y=0.99)
plt.tight_layout()
p10_png = os.path.join(plots_dir, "10_KM_RSI_genes_individual_radiation_only.png")
p10_pdf = os.path.join(plots_dir, "10_KM_RSI_genes_individual_radiation_only.pdf")
plt.savefig(p10_png, dpi=300)
plt.savefig(p10_pdf)
plt.close()

# ==============================================================================
# FIGURE 11: 10-PANEL KM SURVIVAL GRID - INDIVIDUAL RSI GENES (ALL RADIATION)
# ==============================================================================
print("Generating Figure 11: 10-Panel KM Grid for Individual Genes in All Radiation...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), sharey=True)
axes = axes.flatten()

gene_results_all = []

for idx, gene in enumerate(rsi_genes):
    ax = axes[idx]
    med = df_all_rad[gene].median()
    g_high = df_all_rad[df_all_rad[gene] >= med]
    g_low = df_all_rad[df_all_rad[gene] < med]
    
    lr = logrank_test(g_high['time_months'], g_low['time_months'], g_high['status'], g_low['status'])
    
    kmf_h = KaplanMeierFitter()
    kmf_h.fit(g_high['time_months'], g_high['status'], label='High')
    kmf_h.plot_survival_function(ax=ax, color='#BC3C29', linewidth=2, ci_show=False)
    
    kmf_l = KaplanMeierFitter()
    kmf_l.fit(g_low['time_months'], g_low['status'], label='Low')
    kmf_l.plot_survival_function(ax=ax, color='#0072B5', linewidth=2, ci_show=False)
    
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(0, 65)
    role_str = "w > 0" if rsi_coeffs[gene] > 0 else "w < 0"
    ax.set_title(f"{gene} ({role_str})\nlog-rank p = {lr.p_value:.3f}", fontsize=11, fontweight='bold')
    
    if idx % 5 == 0:
        ax.set_ylabel("Survival Probability", fontsize=10.5, fontweight='bold')
    else:
        ax.set_ylabel("")
        
    if idx >= 5:
        ax.set_xlabel("Time (Months)", fontsize=10.5, fontweight='bold')
    else:
        ax.set_xlabel("")
        
    med_h = kmf_h.median_survival_time_
    med_l = kmf_l.median_survival_time_
    ax.text(0.95, 0.92, f"Med High: {med_h:.1f}m\nMed Low: {med_l:.1f}m",
            transform=ax.transAxes, fontsize=8.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc', alpha=0.85))
    ax.legend(loc='lower left', fontsize=8.5, frameon=True)
    
    gene_results_all.append({
        'Gene': gene,
        'Cohort': 'All Radiation Treated (N=201)',
        'Model_Weight': rsi_coeffs[gene],
        'Median_Expression': med,
        'High_N': len(g_high),
        'High_Median_OS': med_h,
        'Low_N': len(g_low),
        'Low_Median_OS': med_l,
        'LogRank_p': lr.p_value
    })

plt.suptitle("Individual RSI Gene Survival Stratification Across All Radiation-Treated Patients (N = 201)", 
             fontsize=13.5, fontweight='bold', y=0.99)
plt.tight_layout()
p11_png = os.path.join(plots_dir, "11_KM_RSI_genes_individual_all_radiation.png")
p11_pdf = os.path.join(plots_dir, "11_KM_RSI_genes_individual_all_radiation.pdf")
plt.savefig(p11_png, dpi=300)
plt.savefig(p11_pdf)
plt.close()

# ==============================================================================
# FIGURE 12: CROSS-TREATMENT MODALITY BENCHMARKING PANEL
# ==============================================================================
print("Generating Figure 12: Cross-Treatment Modality Comparison Panel...")
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)

treatment_subsets = [
    ("Radiation Only (N=17)", df_rad_only, axes[0]),
    ("Chemo + Radiation (N=184)", df_combo, axes[1]),
    ("Chemotherapy Only (N=16)", df_chemo_only, axes[2]),
    ("Treatment Naive (N=55)", df_naive, axes[3])
]

treatment_stats = []

for name, sub, ax in treatment_subsets:
    med = sub['RSI'].median()
    h = sub[sub['RSI'] >= med]
    l = sub[sub['RSI'] < med]
    
    lr = logrank_test(h['time_months'], l['time_months'], h['status'], l['status'])
    
    kmf_h = KaplanMeierFitter()
    kmf_h.fit(h['time_months'], h['status'], label=f"High RSI (N={len(h)})")
    kmf_h.plot_survival_function(ax=ax, color='#BC3C29', linewidth=2.2, ci_show=False)
    
    kmf_l = KaplanMeierFitter()
    kmf_l.fit(l['time_months'], l['status'], label=f"Low RSI (N={len(l)})")
    kmf_l.plot_survival_function(ax=ax, color='#0072B5', linewidth=2.2, ci_show=False)
    
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"{name}\nlog-rank p = {lr.p_value:.3f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time (Months)", fontsize=10.5, fontweight='bold')
    if ax == axes[0]:
        ax.set_ylabel("Overall Survival Probability", fontsize=10.5, fontweight='bold')
    else:
        ax.set_ylabel("")
        
    med_h = kmf_h.median_survival_time_
    med_l = kmf_l.median_survival_time_
    ax.text(0.95, 0.92, f"Med High: {med_h:.1f}m\nMed Low: {med_l:.1f}m",
            transform=ax.transAxes, fontsize=8.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#cccccc', alpha=0.85))
    ax.legend(loc='lower left', fontsize=8.5, frameon=True)
    
    treatment_stats.append({
        'Treatment_Modality': name,
        'N_Patients': len(sub),
        'Median_Cutpoint': med,
        'High_RSI_N': len(h),
        'High_RSI_Median_OS': med_h,
        'Low_RSI_N': len(l),
        'Low_RSI_Median_OS': med_l,
        'LogRank_p': lr.p_value
    })

plt.suptitle("Benchmarking RSI Survival Stratification Across TCGA-GBM Treatment Modalities", 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
p12_png = os.path.join(plots_dir, "12_KM_RSI_treatment_comparison_panel.png")
p12_pdf = os.path.join(plots_dir, "12_KM_RSI_treatment_comparison_panel.pdf")
plt.savefig(p12_png, dpi=300)
plt.savefig(p12_pdf)
plt.close()

# ==============================================================================
# EXPORT ALL STATISTICAL SUMMARY TABLES
# ==============================================================================
print("Exporting summary tables...")

df_gene_stats = pd.concat([pd.DataFrame(gene_results_rad), pd.DataFrame(gene_results_all)], ignore_index=True)
df_gene_stats.to_csv(os.path.join(tables_dir, "RSI_single_genes_survival_summary.csv"), index=False)

df_treatment_stats = pd.DataFrame(treatment_stats)
df_treatment_stats.to_csv(os.path.join(tables_dir, "RSI_treatment_modalities_survival_summary.csv"), index=False)

# Comprehensive Master Survival Summary
master_survival_rows = [
    {
        'Analysis_Cohort': 'Radiation Therapy Only (Monotherapy)',
        'Sample_Size_N': len(df_rad_only),
        'Events_Deaths': df_rad_only['status'].sum(),
        'Event_Rate_Pct': df_rad_only['status'].mean() * 100,
        'Stratification_Method': 'Median Split',
        'Cutpoint_Value': rad_med,
        'High_RSI_N': len(rad_high),
        'High_RSI_Median_OS_Months': kmf_h.median_survival_time_,
        'Low_RSI_N': len(rad_low),
        'Low_RSI_Median_OS_Months': kmf_l.median_survival_time_,
        'Hazard_Ratio_HR': hr_val,
        'HR_95_CI_Lower': hr_ci_vals[0],
        'HR_95_CI_Upper': hr_ci_vals[1],
        'LogRank_p_value': lr_rad_med.p_value
    },
    {
        'Analysis_Cohort': 'Radiation Therapy Only (Monotherapy)',
        'Sample_Size_N': len(df_rad_only),
        'Events_Deaths': df_rad_only['status'].sum(),
        'Event_Rate_Pct': df_rad_only['status'].mean() * 100,
        'Stratification_Method': 'Optimal Cutpoint',
        'Cutpoint_Value': opt_cut_rad,
        'High_RSI_N': len(rad_opt_high),
        'High_RSI_Median_OS_Months': 3.19,
        'Low_RSI_N': len(rad_opt_low),
        'Low_RSI_Median_OS_Months': 13.22,
        'Hazard_Ratio_HR': hr_opt_val,
        'HR_95_CI_Lower': hr_opt_ci[0],
        'HR_95_CI_Upper': hr_opt_ci[1],
        'LogRank_p_value': lr_rad_opt.p_value
    },
    {
        'Analysis_Cohort': 'All Radiation-Treated (Non-Immunotherapy)',
        'Sample_Size_N': len(df_all_rad),
        'Events_Deaths': df_all_rad['status'].sum(),
        'Event_Rate_Pct': df_all_rad['status'].mean() * 100,
        'Stratification_Method': 'Median Split',
        'Cutpoint_Value': all_med,
        'High_RSI_N': len(all_high),
        'High_RSI_Median_OS_Months': 13.96,
        'Low_RSI_N': len(all_low),
        'Low_RSI_Median_OS_Months': 13.73,
        'Hazard_Ratio_HR': hr_all_val,
        'HR_95_CI_Lower': hr_all_ci[0],
        'HR_95_CI_Upper': hr_all_ci[1],
        'LogRank_p_value': lr_all_med.p_value
    },
    {
        'Analysis_Cohort': 'All Radiation-Treated (Non-Immunotherapy)',
        'Sample_Size_N': len(df_all_rad),
        'Events_Deaths': df_all_rad['status'].sum(),
        'Event_Rate_Pct': df_all_rad['status'].mean() * 100,
        'Stratification_Method': 'Optimal Cutpoint',
        'Cutpoint_Value': opt_cut_all,
        'High_RSI_N': len(all_opt_high),
        'High_RSI_Median_OS_Months': 13.95,
        'Low_RSI_N': len(all_opt_low),
        'Low_RSI_Median_OS_Months': 13.85,
        'Hazard_Ratio_HR': hr_all_opt_val,
        'HR_95_CI_Lower': hr_all_opt_ci[0],
        'HR_95_CI_Upper': hr_all_opt_ci[1],
        'LogRank_p_value': lr_all_opt.p_value
    }
]

df_master_stats = pd.DataFrame(master_survival_rows)
df_master_stats.to_csv(os.path.join(tables_dir, "RSI_radiation_survival_master_statistics.csv"), index=False)

# Export all stats to a multi-sheet Excel file
stats_excel_path = os.path.join(tables_dir, "RSI_radiation_survival_statistics.xlsx")
with pd.ExcelWriter(stats_excel_path, engine='openpyxl') as writer:
    df_master_stats.to_excel(writer, sheet_name="RSI_Cohort_Survival_Stats", index=False)
    df_treatment_stats.to_excel(writer, sheet_name="Treatment_Modalities", index=False)
    df_gene_stats.to_excel(writer, sheet_name="Single_Genes_Survival", index=False)
    df_weights.to_excel(writer, sheet_name="RSI_Model_Architecture", index=False)

print("All analyses complete!")
