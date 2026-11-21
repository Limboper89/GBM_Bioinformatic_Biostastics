"""
Script: Strict 11-Gene Serine Metabolism Signature vs. RSI Correlation Analysis
Date: October 2026
Gene Set (Strict 11 Genes):
  PHGDH, PSAT1, PSPH, SHMT1, SHMT2, SLC1A4, SLC1A5, SLC12A4, SLC38A1, SLC38A2, SLC38A5
(MTHFD1 and MTHFD2 completely excluded)
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Style setup
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

# Define STRICT 11-gene signature
serine_11_genes = [
    "PHGDH", "PSAT1", "PSPH", "SHMT1", "SHMT2",
    "SLC1A4", "SLC1A5", "SLC12A4",
    "SLC38A1", "SLC38A2", "SLC38A5"
]

# Load master tables
master_rsi_path = os.path.join(tables_dir, "TCGA_GBM_RSI_patient_master.csv")
df_11_path = os.path.join(work_dir, "TIGER_analysis", "treatment_segregated_analysis", "tcga_gbm_patient_treatment_11gene_serine_detailed.csv")

df_master = pd.read_csv(master_rsi_path)
df_11 = pd.read_csv(df_11_path)

# Drop any MTHFD columns if present
cols_to_drop = [c for c in df_master.columns if 'MTHFD' in c]
if cols_to_drop:
    df_master.drop(columns=cols_to_drop, inplace=True)

# Merge strict 11-gene data
cols_to_add = ['patient_barcode', 'Serine_Score_11'] + serine_11_genes
df_11_clean = df_11[cols_to_add].copy()
df_11_clean.rename(columns={'Serine_Score_11': 'Serine_Score_11gene'}, inplace=True)

# Clean and merge
merge_cols = [c for c in df_master.columns if c not in serine_11_genes and c != 'Serine_Score_11gene']
df_merged = pd.merge(df_master[merge_cols], df_11_clean, on='patient_barcode', how='inner')

# Assign 11-gene Serine_Score
df_merged['Serine_Score'] = df_merged['Serine_Score_11gene']

print(f"Loaded merged cohort: N = {len(df_merged)} patients")
print(f"Strict 11 genes: {serine_11_genes}")

# Subsets
df_rad_only = df_merged[df_merged['treatment_group'] == 'Radiation Only'].copy()
df_all_rad = df_merged[(df_merged['has_rad'] == True) & (df_merged['has_immuno'] == False)].copy()
df_combo = df_merged[df_merged['treatment_group'] == 'Combination (Chemo + Radiation)'].copy()
df_naive = df_merged[df_merged['treatment_group'] == 'Treatment Naive / No Treatment'].copy()
df_chemo_only = df_merged[df_merged['treatment_group'] == 'Chemotherapy Only'].copy()

# ==============================================================================
# FIGURE 22: PRIMARY RSI VS. 11-GENE SERINE SCORE SCATTER PLOT
# ==============================================================================
print("Generating Figure 22: Primary RSI vs 11-Gene Serine Score Scatter Plot...")
fig, ax = plt.subplots(figsize=(8.5, 6.8))

r_all, p_all = stats.pearsonr(df_merged['Serine_Score'], df_merged['RSI'])
rho_all, sp_all = stats.spearmanr(df_merged['Serine_Score'], df_merged['RSI'])
slope, intercept, _, _, _ = stats.linregress(df_merged['Serine_Score'], df_merged['RSI'])

r_ro, p_ro = stats.pearsonr(df_rad_only['Serine_Score'], df_rad_only['RSI'])
rho_ro, sp_ro = stats.spearmanr(df_rad_only['Serine_Score'], df_rad_only['RSI'])

treatment_palette = {
    'Combination (Chemo + Radiation)': '#2b7bba',
    'Radiation Only': '#e41a1c',
    'Treatment Naive / No Treatment': '#4daf4a',
    'Chemotherapy Only': '#984ea3',
    'Immunotherapy': '#ff7f00'
}

for trt, color in treatment_palette.items():
    sub = df_merged[df_merged['treatment_group'] == trt]
    if len(sub) > 0:
        if trt == 'Radiation Only':
            ax.scatter(sub['Serine_Score'], sub['RSI'], color=color, label=f"{trt} (N={len(sub)})",
                       s=95, edgecolors='black', linewidth=1.3, zorder=5, alpha=0.95)
        else:
            ax.scatter(sub['Serine_Score'], sub['RSI'], color=color, label=f"{trt} (N={len(sub)})",
                       s=50, edgecolors='none', alpha=0.65, zorder=3)

sns.regplot(data=df_merged, x='Serine_Score', y='RSI', scatter=False, ax=ax,
            color='#333333', line_kws={'linewidth': 2.2, 'linestyle': '--', 'zorder': 4}, ci=95)

stats_box = (
    f"All Patients (N = {len(df_merged)}):\n"
    f"  Pearson r = {r_all:+.3f} (p = {p_all:.4f})\n"
    f"  Spearman \u03c1 = {rho_all:+.3f} (p = {sp_all:.4f})\n"
    f"  RSI = {intercept:.3f} + ({slope:.3f} \u00d7 Serine Score)\n\n"
    f"Radiation Only (N = {len(df_rad_only)}):\n"
    f"  Pearson r = {r_ro:+.3f} (p = {p_ro:.4f})\n"
    f"  Spearman \u03c1 = {rho_ro:+.3f} (p = {sp_ro:.4f})"
)
ax.text(0.04, 0.96, stats_box, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.92))

ax.set_xlabel("11-Gene Serine Metabolism Score (ssGSEA)", fontsize=11.5, fontweight='bold', labelpad=8)
ax.set_ylabel("Radiosensitivity Index (RSI, Torres-Roca Rank Formula)", fontsize=11.5, fontweight='bold', labelpad=8)
ax.set_title("Correlation: Radiosensitivity Index (RSI) vs. 11-Gene Serine Score in TCGA-GBM\n(Core 8 Genes + SLC38A1/A2/A5 | N = 280)",
             fontsize=12, fontweight='bold', pad=12)

ax.legend(loc='lower left', frameon=True, fontsize=9.5, title="Treatment Cohort", title_fontsize=10)
plt.tight_layout()

f22_png = os.path.join(plots_dir, "22_RSI_vs_11Gene_Serine_Score_primary.png")
f22_pdf = os.path.join(plots_dir, "22_RSI_vs_11Gene_Serine_Score_primary.pdf")
plt.savefig(f22_png, dpi=300)
plt.savefig(f22_pdf)
plt.close()

# ==============================================================================
# FIGURE 23: TREATMENT-SEGREGATED 4-PANEL GRID
# ==============================================================================
print("Generating Figure 23: Treatment-Segregated 4-Panel Grid...")
fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
axes = axes.flatten()

panels_info = [
    ("A. Radiation Therapy Only (Monotherapy)", df_rad_only, '#e41a1c', axes[0]),
    ("B. All Radiation-Treated Non-Immuno", df_all_rad, '#2b7bba', axes[1]),
    ("C. Combination Chemoradiotherapy (Stupp)", df_combo, '#3182bd', axes[2]),
    ("D. Treatment Naive / Resection Only", df_naive, '#4daf4a', axes[3])
]

cohort_corrs = []

for title, sub, col, ax in panels_info:
    pr, pp = stats.pearsonr(sub['Serine_Score'], sub['RSI'])
    sr, sp = stats.spearmanr(sub['Serine_Score'], sub['RSI'])
    sl, ic, _, _, _ = stats.linregress(sub['Serine_Score'], sub['RSI'])
    
    ax.scatter(sub['Serine_Score'], sub['RSI'], color=col, s=70, alpha=0.75,
               edgecolors='black', linewidth=0.8, zorder=3)
    
    sns.regplot(data=sub, x='Serine_Score', y='RSI', scatter=False, ax=ax,
                color=col, line_kws={'linewidth': 2.2, 'zorder': 4}, ci=95)
    
    ax.set_title(f"{title} (N = {len(sub)})", fontsize=11.5, fontweight='bold', pad=8)
    ax.set_xlabel("11-Gene Serine Score (ssGSEA)", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("RSI Score", fontsize=10.5, fontweight='bold')
    
    p_text = f"p = {pp:.4f}" if pp >= 0.001 else f"p = {pp:.2e}"
    sp_text = f"p = {sp:.4f}" if sp >= 0.001 else f"p = {sp:.2e}"
    
    stat_str = (
        f"Pearson r = {pr:+.3f} ({p_text})\n"
        f"Spearman \u03c1 = {sr:+.3f} ({sp_text})\n"
        f"Slope = {sl:+.3f}"
    )
    ax.text(0.05, 0.94, stat_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    
    cohort_corrs.append({
        'Cohort': title,
        'Sample_Size_N': len(sub),
        'Pearson_r': pr,
        'Pearson_p_value': pp,
        'Spearman_rho': sr,
        'Spearman_p_value': sp,
        'Slope': sl,
        'Intercept': ic
    })

plt.suptitle("11-Gene Serine Score vs. RSI Across Clinical Treatment Cohorts in TCGA-GBM",
             fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()

f23_png = os.path.join(plots_dir, "23_RSI_vs_11Gene_Serine_Score_treatment_grid.png")
f23_pdf = os.path.join(plots_dir, "23_RSI_vs_11Gene_Serine_Score_treatment_grid.pdf")
plt.savefig(f23_png, dpi=300)
plt.savefig(f23_pdf)
plt.close()

# ==============================================================================
# FIGURE 24: LOLLIPOP CHART FOR STRICT 11 GENES + COMPOSITE SCORE
# ==============================================================================
print("Generating Figure 24: Strict 11 Genes + Serine Score Lollipop Chart...")
gene_features = serine_11_genes + ['Serine_Score']

gene_corrs = []
for g in gene_features:
    r_all, p_all = stats.pearsonr(df_merged['RSI'], df_merged[g])
    rho_all, sp_all = stats.spearmanr(df_merged['RSI'], df_merged[g])
    
    r_rad, p_rad = stats.pearsonr(df_rad_only['RSI'], df_rad_only[g])
    rho_rad, sp_rad = stats.spearmanr(df_rad_only['RSI'], df_rad_only[g])
    
    r_allrad, p_allrad = stats.pearsonr(df_all_rad['RSI'], df_all_rad[g])
    rho_allrad, sp_allrad = stats.spearmanr(df_all_rad['RSI'], df_all_rad[g])
    
    gene_corrs.append({
        'Feature': g,
        'Pearson_r_All': r_all, 'Pearson_p_All': p_all, 'Spearman_rho_All': rho_all, 'Spearman_p_All': sp_all,
        'Pearson_r_RadOnly': r_rad, 'Pearson_p_RadOnly': p_rad, 'Spearman_rho_RadOnly': rho_rad, 'Spearman_p_RadOnly': sp_rad,
        'Pearson_r_AllRad': r_allrad, 'Pearson_p_AllRad': p_allrad, 'Spearman_rho_AllRad': rho_allrad, 'Spearman_p_AllRad': sp_allrad
    })

df_gene_corrs = pd.DataFrame(gene_corrs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

df_pop = df_gene_corrs.sort_values('Spearman_rho_All', ascending=True).copy()
y_pos = np.arange(len(df_pop))

# Panel 1: All Patients
colors_all = ['#2b7bba' if r < 0 else '#d95f02' for r in df_pop['Spearman_rho_All']]
ax1.hlines(y=y_pos, xmin=0, xmax=df_pop['Spearman_rho_All'], color=colors_all, alpha=0.7, linewidth=2.5)
ax1.scatter(df_pop['Spearman_rho_All'], y_pos, color=colors_all, s=140, edgecolors='black', linewidth=1.2, zorder=3)

for idx, (rho, p, feat) in enumerate(zip(df_pop['Spearman_rho_All'], df_pop['Spearman_p_All'], df_pop['Feature'])):
    sig = "****" if p < 0.0001 else ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")))
    offset = 0.02 if rho >= 0 else -0.02
    ha = 'left' if rho >= 0 else 'right'
    font_wt = "bold" if sig != "ns" else "normal"
    ax1.text(rho + offset, idx, f"{rho:+.3f} ({sig})", va='center', ha=ha, fontsize=9.5, fontweight=font_wt)

ax1.axvline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_pop['Feature'], fontsize=11, fontweight='bold')
ax1.set_xlim(-0.45, 0.45)
ax1.set_xlabel("Spearman Correlation (\u03c1)", fontsize=11, fontweight='bold')
ax1.set_title("A. All TCGA-GBM Patients (N = 280)", fontsize=12, fontweight='bold')

# Panel 2: Radiation Only
colors_rad = ['#2b7bba' if r < 0 else '#d95f02' for r in df_pop['Spearman_rho_RadOnly']]
ax2.hlines(y=y_pos, xmin=0, xmax=df_pop['Spearman_rho_RadOnly'], color=colors_rad, alpha=0.7, linewidth=2.5)
ax2.scatter(df_pop['Spearman_rho_RadOnly'], y_pos, color=colors_rad, s=140, edgecolors='black', linewidth=1.2, zorder=3)

for idx, (rho, p, feat) in enumerate(zip(df_pop['Spearman_rho_RadOnly'], df_pop['Spearman_p_RadOnly'], df_pop['Feature'])):
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("\u2020" if p < 0.10 else "ns"))
    offset = 0.025 if rho >= 0 else -0.025
    ha = 'left' if rho >= 0 else 'right'
    font_wt = "bold" if sig != "ns" else "normal"
    ax2.text(rho + offset, idx, f"{rho:+.3f} ({sig})", va='center', ha=ha, fontsize=9.5, fontweight=font_wt)

ax2.axvline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlim(-0.65, 0.65)
ax2.set_xlabel("Spearman Correlation (\u03c1)", fontsize=11, fontweight='bold')
ax2.set_title("B. Radiation Therapy Only Cohort (N = 17)", fontsize=12, fontweight='bold')

plt.suptitle("Correlation of Strict 11-Gene Serine Metabolism Signature vs. Radiosensitivity Index (RSI)",
             fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()

f24_png = os.path.join(plots_dir, "24_RSI_vs_11_Serine_Genes_lollipop.png")
f24_pdf = os.path.join(plots_dir, "24_RSI_vs_11_Serine_Genes_lollipop.pdf")
plt.savefig(f24_png, dpi=300)
plt.savefig(f24_pdf)
plt.close()

# ==============================================================================
# FIGURE 25: 12-PANEL SCATTER GRID FOR ALL 11 GENES + SERINE SCORE VS RSI
# ==============================================================================
print("Generating Figure 25: 12-Panel Scatter Grid for All 11 Genes + Serine Score...")
fig, axes = plt.subplots(3, 4, figsize=(17, 12))
axes = axes.flatten()

grid_features = serine_11_genes + ['Serine_Score']

for idx, feat in enumerate(grid_features):
    ax = axes[idx]
    
    r_val, p_val = stats.pearsonr(df_merged[feat], df_merged['RSI'])
    rho_val, sp_val = stats.spearmanr(df_merged[feat], df_merged['RSI'])
    
    # In rad only
    r_ro, p_ro = stats.pearsonr(df_rad_only[feat], df_rad_only['RSI'])
    
    if feat == 'Serine_Score':
        color_pt = '#e7298a'
        title_str = "Serine Score (11-Gene ssGSEA)"
        xlabel_str = "ssGSEA Enrichment Score"
    elif 'SLC38' in feat:
        color_pt = '#d95f02'
        title_str = f"{feat} (System A/N)"
        xlabel_str = f"{feat} [Log2(FPKM + 1)]"
    elif 'SLC' in feat:
        color_pt = '#7570b3'
        title_str = f"{feat} (Transport)"
        xlabel_str = f"{feat} [Log2(FPKM + 1)]"
    else:
        color_pt = '#2b7bba'
        title_str = f"{feat} (Enzyme)"
        xlabel_str = f"{feat} [Log2(FPKM + 1)]"
        
    sns.regplot(data=df_merged, x=feat, y='RSI', ax=ax, color=color_pt,
                scatter_kws={'alpha': 0.55, 's': 35}, line_kws={'color': '#222222', 'linewidth': 1.8})
    
    ax.set_title(title_str, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel_str, fontsize=9.5, fontweight='bold')
    
    if idx % 4 == 0:
        ax.set_ylabel("RSI Score", fontsize=10, fontweight='bold')
    else:
        ax.set_ylabel("")
        
    sig = "****" if p_val < 0.0001 else ("***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")))
    
    box_txt = (
        f"All (N=280): r = {r_val:+.3f} ({sig})\n"
        f"Rad Only: r = {r_ro:+.3f}"
    )
    ax.text(0.04, 0.94, box_txt, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))

plt.suptitle("Individual Gene Scatters: 11 Serine Pathway Genes & Composite Score vs. RSI (N = 280 TCGA-GBM)",
             fontsize=13.5, fontweight='bold', y=0.995)
plt.tight_layout()

f25_png = os.path.join(plots_dir, "25_RSI_vs_all_11_Serine_genes_scatters.png")
f25_pdf = os.path.join(plots_dir, "25_RSI_vs_all_11_Serine_genes_scatters.pdf")
plt.savefig(f25_png, dpi=300)
plt.savefig(f25_pdf)
plt.close()

# ==============================================================================
# EXPORT DEDICATED TABLES
# ==============================================================================
print("Exporting strict 11-gene correlation tables...")
df_cohort_stats = pd.DataFrame(cohort_corrs)
df_cohort_stats.to_csv(os.path.join(tables_dir, "RSI_vs_11Gene_Serine_Score_cohort_correlations.csv"), index=False)
df_gene_corrs.to_csv(os.path.join(tables_dir, "RSI_vs_11_individual_serine_genes_correlations.csv"), index=False)

# Dedicated Excel workbook
excel_path = os.path.join(tables_dir, "RSI_vs_11Gene_Serine_Score_summary.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_cohort_stats.to_excel(writer, sheet_name="Cohort_Correlations", index=False)
    df_gene_corrs.to_excel(writer, sheet_name="11Genes_Correlations", index=False)

print("Strict 11-gene analysis complete! Outputs saved.")
