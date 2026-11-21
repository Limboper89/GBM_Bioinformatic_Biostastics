"""
Script: RSI vs. Serine Score Correlation Analysis and Publication Plots
Date: October 2026
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="ticks", font_scale=1.1)
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

# Load data
master_path = os.path.join(tables_dir, "TCGA_GBM_RSI_patient_master.csv")
df = pd.read_csv(master_path)
print(f"Loaded master cohort: N = {len(df)} patients")

# Treatment subsets
df_rad_only = df[df['treatment_group'] == 'Radiation Only'].copy()
df_all_rad = df[(df['has_rad'] == True) & (df['has_immuno'] == False)].copy()
df_combo = df[df['treatment_group'] == 'Combination (Chemo + Radiation)'].copy()
df_naive = df[df['treatment_group'] == 'Treatment Naive / No Treatment'].copy()
df_chemo_only = df[df['treatment_group'] == 'Chemotherapy Only'].copy()

# ==============================================================================
# 1. PRIMARY SCATTER PLOT: WHOLE TCGA-GBM COHORT WITH TREATMENT COLORS
# ==============================================================================
print("Generating Primary RSI vs Serine Score Scatter Plot...")
fig, ax = plt.subplots(figsize=(8.5, 6.8))

# Linear regression for whole cohort
slope, intercept, r_val, p_val, std_err = stats.linregress(df['Serine_Score'], df['RSI'])
spearman_rho, spearman_p = stats.spearmanr(df['Serine_Score'], df['RSI'])

treatment_palette = {
    'Combination (Chemo + Radiation)': '#2b7bba',
    'Radiation Only': '#e41a1c',
    'Treatment Naive / No Treatment': '#4daf4a',
    'Chemotherapy Only': '#984ea3',
    'Immunotherapy': '#ff7f00'
}

# Plot points by treatment
for trt, color in treatment_palette.items():
    sub = df[df['treatment_group'] == trt]
    if len(sub) > 0:
        if trt == 'Radiation Only':
            ax.scatter(sub['Serine_Score'], sub['RSI'], color=color, label=f"{trt} (N={len(sub)})",
                       s=90, edgecolors='black', linewidth=1.2, zorder=5, alpha=0.95)
        else:
            ax.scatter(sub['Serine_Score'], sub['RSI'], color=color, label=f"{trt} (N={len(sub)})",
                       s=50, edgecolors='none', alpha=0.65, zorder=3)

# Regression line across all patients
x_vals = np.linspace(df['Serine_Score'].min(), df['Serine_Score'].max(), 100)
y_vals = intercept + slope * x_vals

# Confidence band
sns.regplot(data=df, x='Serine_Score', y='RSI', scatter=False, ax=ax,
            color='#333333', line_kws={'linewidth': 2.2, 'linestyle': '--', 'zorder': 4},
            ci=95)

# Annotation box
stats_box = (
    f"All Patients (N = {len(df)}):\n"
    f"  Pearson r = {r_val:.3f} (p = {p_val:.4f})\n"
    f"  Spearman \u03c1 = {spearman_rho:.3f} (p = {spearman_p:.4f})\n"
    f"  Fit: RSI = {intercept:.3f} + ({slope:.3f} \u00d7 Serine Score)\n\n"
    f"Radiation Only (N = {len(df_rad_only)}):\n"
    f"  Pearson r = -0.466 (p = 0.0594)\n"
    f"  Spearman \u03c1 = -0.431 (p = 0.0838)"
)
ax.text(0.04, 0.96, stats_box, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.92))

# Directional labels
ax.text(df['Serine_Score'].max() * 0.98, df['RSI'].max() * 0.98, "Radioresistant (High RSI) \u2191", 
        fontsize=9, color='#888888', ha='right', va='top', style='italic')
ax.text(df['Serine_Score'].max() * 0.98, df['RSI'].min() * 1.02, "Radiosensitive (Low RSI) \u2193", 
        fontsize=9, color='#888888', ha='right', va='bottom', style='italic')

ax.set_xlabel("Serine Metabolism Score (ssGSEA)", fontsize=11.5, fontweight='bold', labelpad=8)
ax.set_ylabel("Radiosensitivity Index (RSI, Torres-Roca Rank Formula)", fontsize=11.5, fontweight='bold', labelpad=8)
ax.set_title("Correlation: Radiosensitivity Index (RSI) vs. Serine Score in TCGA-GBM\n(Overall Cohort N = 280)",
             fontsize=12.5, fontweight='bold', pad=12)

ax.legend(loc='lower left', frameon=True, fontsize=9.5, title="Treatment Cohort", title_fontsize=10)
plt.tight_layout()

f1_png = os.path.join(plots_dir, "13_RSI_vs_Serine_Score_correlation_primary.png")
f1_pdf = os.path.join(plots_dir, "13_RSI_vs_Serine_Score_correlation_primary.pdf")
plt.savefig(f1_png, dpi=300)
plt.savefig(f1_pdf)
plt.close()

# ==============================================================================
# 2. TREATMENT-SEGREGATED MULTI-PANEL SCATTER GRID (4 PANELS)
# ==============================================================================
print("Generating 4-Panel Treatment Segregated Scatter Grid...")
fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
axes = axes.flatten()

panels_info = [
    ("A. Radiation Therapy Only (Monotherapy)", df_rad_only, '#e41a1c', axes[0]),
    ("B. All Radiation-Treated Non-Immuno", df_all_rad, '#2b7bba', axes[1]),
    ("C. Combination Chemoradiotherapy (Stupp)", df_combo, '#3182bd', axes[2]),
    ("D. Treatment Naive / Resection Only", df_naive, '#4daf4a', axes[3])
]

corr_summary_rows = []

for title, sub, col, ax in panels_info:
    pr, pp = stats.pearsonr(sub['Serine_Score'], sub['RSI'])
    sr, sp = stats.spearmanr(sub['Serine_Score'], sub['RSI'])
    sl, ic, _, _, _ = stats.linregress(sub['Serine_Score'], sub['RSI'])
    
    # Scatter points
    ax.scatter(sub['Serine_Score'], sub['RSI'], color=col, s=70, alpha=0.75,
               edgecolors='black', linewidth=0.8, zorder=3)
    
    # Regression line with CI
    sns.regplot(data=sub, x='Serine_Score', y='RSI', scatter=False, ax=ax,
                color=col, line_kws={'linewidth': 2.2, 'zorder': 4}, ci=95)
    
    ax.set_title(f"{title} (N = {len(sub)})", fontsize=11.5, fontweight='bold', pad=8)
    ax.set_xlabel("Serine Score (ssGSEA)", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("RSI Score", fontsize=10.5, fontweight='bold')
    
    # Annotate stats
    p_text = f"p = {pp:.4f}" if pp >= 0.001 else f"p = {pp:.2e}"
    sp_text = f"p = {sp:.4f}" if sp >= 0.001 else f"p = {sp:.2e}"
    sig_mark = " *" if pp < 0.05 else (" \u2020" if pp < 0.10 else " (ns)")
    
    stat_str = (
        f"Pearson r = {pr:.3f}{sig_mark} ({p_text})\n"
        f"Spearman \u03c1 = {sr:.3f} ({sp_text})\n"
        f"Slope = {sl:.3f}"
    )
    ax.text(0.05, 0.94, stat_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    
    corr_summary_rows.append({
        'Cohort': title,
        'Sample_Size_N': len(sub),
        'Pearson_r': pr,
        'Pearson_p_value': pp,
        'Spearman_rho': sr,
        'Spearman_p_value': sp,
        'Slope': sl,
        'Intercept': ic
    })

plt.suptitle("Radiosensitivity Index (RSI) vs. Serine Score Across Clinical Treatment Cohorts",
             fontsize=13.5, fontweight='bold', y=0.995)
plt.tight_layout()

f2_png = os.path.join(plots_dir, "14_RSI_vs_Serine_Score_treatment_panel.png")
f2_pdf = os.path.join(plots_dir, "14_RSI_vs_Serine_Score_treatment_panel.pdf")
plt.savefig(f2_png, dpi=300)
plt.savefig(f2_pdf)
plt.close()

# ==============================================================================
# 3. JOINT DENSITY AND MARGINAL DISTRIBUTION PLOT (SEABORN JOINTPLOT)
# ==============================================================================
print("Generating Joint Distribution & KDE Contour Plot...")
g = sns.jointplot(
    data=df, x='Serine_Score', y='RSI', kind='reg',
    color='#1f77b4', height=7, ratio=4,
    scatter_kws={'alpha': 0.6, 's': 45, 'edgecolor': 'none'},
    line_kws={'color': '#d62728', 'linewidth': 2.2},
    marginal_kws={'fill': True, 'color': '#1f77b4', 'alpha': 0.4}
)
g.plot_joint(sns.kdeplot, color='#d62728', zorder=2, levels=5, alpha=0.5, linewidths=1.2)

g.ax_joint.set_xlabel("Serine Score (ssGSEA)", fontsize=11, fontweight='bold')
g.ax_joint.set_ylabel("Radiosensitivity Index (RSI)", fontsize=11, fontweight='bold')

g.fig.suptitle("Bivariate Distribution & Density: RSI vs. Serine Score (N = 280)\n"
               f"Pearson r = {r_val:.3f} (p = {p_val:.4f}) | Spearman \u03c1 = {spearman_rho:.3f} (p = {spearman_p:.4f})",
               fontsize=11.5, fontweight='bold', y=1.02)

f3_png = os.path.join(plots_dir, "15_RSI_vs_Serine_Score_jointplot.png")
f3_pdf = os.path.join(plots_dir, "15_RSI_vs_Serine_Score_jointplot.pdf")
plt.savefig(f3_png, dpi=300)
plt.savefig(f3_pdf)
plt.close()

# ==============================================================================
# 4. LOLLIPOP PLOT: INDIVIDUAL SERINE GENES CORRELATION WITH RSI
# ==============================================================================
print("Generating Serine Pathway Genes vs RSI Lollipop Plot...")
serine_genes = ['PHGDH', 'PSAT1', 'PSPH', 'SHMT1', 'SHMT2', 'SLC1A4', 'SLC1A5', 'SLC12A4', 'Serine_Score']

gene_corrs = []
for g in serine_genes:
    # In All patients
    r_all, p_all = stats.pearsonr(df['RSI'], df[g])
    rho_all, sp_all = stats.spearmanr(df['RSI'], df[g])
    # In Rad Only
    r_rad, p_rad = stats.pearsonr(df_rad_only['RSI'], df_rad_only[g])
    rho_rad, sp_rad = stats.spearmanr(df_rad_only['RSI'], df_rad_only[g])
    # In All Rad
    r_allrad, p_allrad = stats.pearsonr(df_all_rad['RSI'], df_all_rad[g])
    rho_allrad, sp_allrad = stats.spearmanr(df_all_rad['RSI'], df_all_rad[g])
    
    gene_corrs.append({
        'Gene_or_Score': g,
        'Pearson_r_All': r_all, 'Pearson_p_All': p_all, 'Spearman_rho_All': rho_all, 'Spearman_p_All': sp_all,
        'Pearson_r_RadOnly': r_rad, 'Pearson_p_RadOnly': p_rad, 'Spearman_rho_RadOnly': rho_rad, 'Spearman_p_RadOnly': sp_rad,
        'Pearson_r_AllRad': r_allrad, 'Pearson_p_AllRad': p_allrad, 'Spearman_rho_AllRad': rho_allrad, 'Spearman_p_AllRad': sp_allrad
    })

df_gene_corrs = pd.DataFrame(gene_corrs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Sort by Spearman rho in All Patients
df_lollipop = df_gene_corrs.sort_values('Spearman_rho_All', ascending=True).copy()
y_pos = np.arange(len(df_lollipop))

# Panel 1: All Patients
colors_all = ['#2b7bba' if r < 0 else '#d95f02' for r in df_lollipop['Spearman_rho_All']]
ax1.hlines(y=y_pos, xmin=0, xmax=df_lollipop['Spearman_rho_All'], color=colors_all, alpha=0.7, linewidth=2.5)
ax1.scatter(df_lollipop['Spearman_rho_All'], y_pos, color=colors_all, s=130, edgecolors='black', linewidth=1.2, zorder=3)

for idx, (rho, p) in enumerate(zip(df_lollipop['Spearman_rho_All'], df_lollipop['Spearman_p_All'])):
    sig = "****" if p < 0.0001 else ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")))
    offset = 0.02 if rho >= 0 else -0.02
    ha = 'left' if rho >= 0 else 'right'
    ax1.text(rho + offset, idx, f"{rho:+.3f} ({sig})", va='center', ha=ha, fontsize=9.5, fontweight='bold')

ax1.axvline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_lollipop['Gene_or_Score'], fontsize=11, fontweight='bold')
ax1.set_xlim(-0.45, 0.45)
ax1.set_xlabel("Spearman Correlation (\u03c1)", fontsize=11, fontweight='bold')
ax1.set_title("A. All TCGA-GBM Patients (N = 280)", fontsize=11.5, fontweight='bold')

# Panel 2: Radiation Only
colors_rad = ['#2b7bba' if r < 0 else '#d95f02' for r in df_lollipop['Spearman_rho_RadOnly']]
ax2.hlines(y=y_pos, xmin=0, xmax=df_lollipop['Spearman_rho_RadOnly'], color=colors_rad, alpha=0.7, linewidth=2.5)
ax2.scatter(df_lollipop['Spearman_rho_RadOnly'], y_pos, color=colors_rad, s=130, edgecolors='black', linewidth=1.2, zorder=3)

for idx, (rho, p) in enumerate(zip(df_lollipop['Spearman_rho_RadOnly'], df_lollipop['Spearman_p_RadOnly'])):
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("\u2020" if p < 0.10 else "ns"))
    offset = 0.025 if rho >= 0 else -0.025
    ha = 'left' if rho >= 0 else 'right'
    ax2.text(rho + offset, idx, f"{rho:+.3f} ({sig})", va='center', ha=ha, fontsize=9.5, fontweight='bold')

ax2.axvline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlim(-0.65, 0.65)
ax2.set_xlabel("Spearman Correlation (\u03c1)", fontsize=11, fontweight='bold')
ax2.set_title("B. Radiation Therapy Only Cohort (N = 17)", fontsize=11.5, fontweight='bold')

plt.suptitle("Correlation of Individual Serine Pathway Genes vs. RSI Score", fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout()

f4_png = os.path.join(plots_dir, "16_RSI_vs_Serine_genes_lollipop.png")
f4_pdf = os.path.join(plots_dir, "16_RSI_vs_Serine_genes_lollipop.pdf")
plt.savefig(f4_png, dpi=300)
plt.savefig(f4_pdf)
plt.close()

# ==============================================================================
# 5. MULTI-METRIC COMPARISON (RANK RSI, LOG2-FPKM RSI, LOG2-TPM RSI)
# ==============================================================================
print("Generating Multi-Metric RSI vs Serine Score Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

rsi_metrics = [
    ("RSI (Rank-based)", "RSI", axes[0]),
    ("RSI (Continuous Log2-FPKM)", "RSI_log2_fpkm", axes[1]),
    ("RSI (Continuous Log2-TPM)", "RSI_log2_tpm", axes[2])
]

for label, col_name, ax in rsi_metrics:
    r_val, p_val = stats.pearsonr(df['Serine_Score'], df[col_name])
    rho_val, sp_val = stats.spearmanr(df['Serine_Score'], df[col_name])
    
    sns.regplot(data=df, x='Serine_Score', y=col_name, ax=ax, color='#1f77b4',
                scatter_kws={'alpha': 0.6, 's': 45}, line_kws={'color': '#d62728', 'linewidth': 2})
    
    ax.set_title(f"{label}\nr = {r_val:.3f} (p = {p_val:.4f}) | \u03c1 = {rho_val:.3f}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Serine Score (ssGSEA)", fontsize=10.5, fontweight='bold')
    ax.set_ylabel(f"{label} Score", fontsize=10.5, fontweight='bold')

plt.suptitle("Robustness Across RSI Normalization Formulations (N = 280 TCGA-GBM)", fontsize=12.5, fontweight='bold', y=1.02)
plt.tight_layout()

f5_png = os.path.join(plots_dir, "17_RSI_formulations_vs_Serine_Score.png")
f5_pdf = os.path.join(plots_dir, "17_RSI_formulations_vs_Serine_Score.pdf")
plt.savefig(f5_png, dpi=300)
plt.savefig(f5_pdf)
plt.close()

# ==============================================================================
# EXPORT STATISTICAL DATA TABLES
# ==============================================================================
print("Exporting correlation statistics tables...")
df_cohort_corrs = pd.DataFrame(corr_summary_rows)
df_cohort_corrs.to_csv(os.path.join(tables_dir, "RSI_vs_Serine_Score_cohort_correlations.csv"), index=False)
df_gene_corrs.to_csv(os.path.join(tables_dir, "RSI_vs_Serine_genes_correlations.csv"), index=False)

# Excel workbook
excel_corr_path = os.path.join(tables_dir, "RSI_vs_Serine_Score_correlation_summary.xlsx")
with pd.ExcelWriter(excel_corr_path, engine='openpyxl') as writer:
    df_cohort_corrs.to_excel(writer, sheet_name="Cohort_Correlations", index=False)
    df_gene_corrs.to_excel(writer, sheet_name="Serine_Genes_Correlations", index=False)

print("All correlation analyses complete! Figures and tables saved.")
