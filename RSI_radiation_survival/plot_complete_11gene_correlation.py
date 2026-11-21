"""
Script: Complete 11-Gene Serine Signature (including SLC38A1, SLC38A2, SLC38A5) vs. RSI Correlation Analysis
Date: October 2026
"""

import os
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
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

# Load files
master_rsi_path = os.path.join(tables_dir, "TCGA_GBM_RSI_patient_master.csv")
df_11_path = os.path.join(work_dir, "TIGER_analysis", "treatment_segregated_analysis", "tcga_gbm_patient_treatment_11gene_serine_detailed.csv")

df_master = pd.read_csv(master_rsi_path)
df_11 = pd.read_csv(df_11_path)

# Merge 11-gene data into master table
cols_to_add = ['patient_barcode', 'Serine_Score_11', 'Serine_Score_8', 'SLC38A1', 'SLC38A2', 'SLC38A5', 'MTHFD1', 'MTHFD2']
for c in cols_to_add[1:]:
    if c in df_master.columns:
        df_master.drop(columns=[c], inplace=True)

df_merged = pd.merge(df_master, df_11[cols_to_add], on='patient_barcode', how='inner')

# Save updated master table
df_merged.to_csv(master_rsi_path, index=False)
with pd.ExcelWriter(os.path.join(tables_dir, "TCGA_GBM_RSI_patient_master.xlsx"), engine='openpyxl') as writer:
    df_merged.to_excel(writer, sheet_name="Master_Patient_Cohort", index=False)
    df_merged[df_merged['treatment_group'] == 'Radiation Only'].to_excel(writer, sheet_name="Radiation_Only_Cohort", index=False)
    df_merged[(df_merged['has_rad'] == True) & (df_merged['has_immuno'] == False)].to_excel(writer, sheet_name="All_Rad_Non_Immuno", index=False)

print(f"Updated master dataset with complete 11-gene signature: N = {len(df_merged)} patients")

df_rad_only = df_merged[df_merged['treatment_group'] == 'Radiation Only'].copy()
df_all_rad = df_merged[(df_merged['has_rad'] == True) & (df_merged['has_immuno'] == False)].copy()
df_combo = df_merged[df_merged['treatment_group'] == 'Combination (Chemo + Radiation)'].copy()
df_naive = df_merged[df_merged['treatment_group'] == 'Treatment Naive / No Treatment'].copy()

# ==============================================================================
# FIGURE 18: SIDE-BY-SIDE 8-GENE VS. 11-GENE SERINE SCORE VS. RSI
# ==============================================================================
print("Generating Figure 18: 8-Gene vs 11-Gene Serine Score Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Panel 1: 8-Gene Score
r8, p8 = stats.pearsonr(df_merged['Serine_Score_8'], df_merged['RSI'])
rho8, sp8 = stats.spearmanr(df_merged['Serine_Score_8'], df_merged['RSI'])
sl8, ic8, _, _, _ = stats.linregress(df_merged['Serine_Score_8'], df_merged['RSI'])

sns.regplot(data=df_merged, x='Serine_Score_8', y='RSI', ax=ax1,
            color='#2b7bba', scatter_kws={'alpha': 0.6, 's': 55}, line_kws={'color': '#d95f02', 'linewidth': 2.2})
ax1.set_title("A. Core 8-Gene Serine Score vs. RSI\n(Without SLC38 Family)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Core 8-Gene Serine Score (ssGSEA)", fontsize=11, fontweight='bold')
ax1.set_ylabel("Radiosensitivity Index (RSI)", fontsize=11, fontweight='bold')

stat_box_8 = (
    f"Pearson r = {r8:.3f} (p = {p8:.4f}) *\n"
    f"Spearman \u03c1 = {rho8:.3f} (p = {sp8:.4f}) *\n"
    f"Slope = {sl8:.3f}\n"
    "Significant Negative Correlation:\n"
    "High 8-Gene Serine \u2192 Lower RSI (Radiosensitive)"
)
ax1.text(0.04, 0.95, stat_box_8, transform=ax1.transAxes, fontsize=9.5,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

# Panel 2: 11-Gene Score
r11, p11 = stats.pearsonr(df_merged['Serine_Score_11'], df_merged['RSI'])
rho11, sp11 = stats.spearmanr(df_merged['Serine_Score_11'], df_merged['RSI'])
sl11, ic11, _, _, _ = stats.linregress(df_merged['Serine_Score_11'], df_merged['RSI'])

sns.regplot(data=df_merged, x='Serine_Score_11', y='RSI', ax=ax2,
            color='#7570b3', scatter_kws={'alpha': 0.6, 's': 55}, line_kws={'color': '#e7298a', 'linewidth': 2.2})
ax2.set_title("B. Expanded 11-Gene Serine Score vs. RSI\n(Incorporating SLC38A1, SLC38A2, SLC38A5)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Expanded 11-Gene Serine Score (ssGSEA)", fontsize=11, fontweight='bold')
ax2.set_ylabel("")

stat_box_11 = (
    f"Pearson r = {r11:.3f} (p = {p11:.4f}) (ns)\n"
    f"Spearman \u03c1 = {rho11:.3f} (p = {sp11:.4f}) (ns)\n"
    f"Slope = {sl11:.3f}\n"
    "Neutralized by SLC38A1:\n"
    "Strong positive SLC38A1 (r = +0.340) offsets\n"
    "core serine synthesis negative coupling"
)
ax2.text(0.04, 0.95, stat_box_11, transform=ax2.transAxes, fontsize=9.5,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))

plt.suptitle("Comparative Impact of Signature Expansion: Core 8-Gene vs. Expanded 11-Gene Serine Score vs. RSI (N = 280)",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

f18_png = os.path.join(plots_dir, "18_RSI_vs_Serine_Score_8_vs_11_comparison.png")
f18_pdf = os.path.join(plots_dir, "18_RSI_vs_Serine_Score_8_vs_11_comparison.pdf")
plt.savefig(f18_png, dpi=300)
plt.savefig(f18_pdf)
plt.close()

# ==============================================================================
# FIGURE 19: DEDICATED SLC38A FAMILY VS. RSI SCATTER PLOTS
# ==============================================================================
print("Generating Figure 19: SLC38A Family vs RSI Scatters...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)

slc38_genes = [
    ("SLC38A1", "SNAT1 Neutral AA Transporter", "#d95f02", axes[0]),
    ("SLC38A2", "SNAT2 Stress-Induced Transporter", "#7570b3", axes[1]),
    ("SLC38A5", "SNAT5 System N Transporter", "#1b9e77", axes[2])
]

slc_stats = []

for g, full_name, col, ax in slc38_genes:
    r_val, p_val = stats.pearsonr(df_merged[g], df_merged['RSI'])
    rho_val, sp_val = stats.spearmanr(df_merged[g], df_merged['RSI'])
    slope, intercept, _, _, _ = stats.linregress(df_merged[g], df_merged['RSI'])
    
    # In All Radiation (N=201)
    r_rad, p_rad = stats.pearsonr(df_all_rad[g], df_all_rad['RSI'])
    
    # In Rad Only (N=17)
    r_ro, p_ro = stats.pearsonr(df_rad_only[g], df_rad_only['RSI'])
    
    sns.regplot(data=df_merged, x=g, y='RSI', ax=ax, color=col,
                scatter_kws={'alpha': 0.6, 's': 55, 'edgecolors': 'black', 'linewidths': 0.5},
                line_kws={'color': '#333333', 'linewidth': 2.2})
    
    ax.set_title(f"{g} ({full_name})", fontsize=11.5, fontweight='bold', pad=8)
    ax.set_xlabel(f"{g} Expression [Log2(FPKM + 1)]", fontsize=10.5, fontweight='bold')
    if ax == axes[0]:
        ax.set_ylabel("Radiosensitivity Index (RSI)", fontsize=11, fontweight='bold')
    else:
        ax.set_ylabel("")
        
    sig_str = "****" if p_val < 0.0001 else ("***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns")))
    
    stat_text = (
        f"All Patients (N = 280):\n"
        f"  Pearson r = {r_val:+.3f} ({sig_str})\n"
        f"  p = {p_val:.2e}\n"
        f"  Spearman \u03c1 = {rho_val:+.3f} (p = {sp_val:.2e})\n\n"
        f"All Rad-Treated (N = 201):\n"
        f"  Pearson r = {r_rad:+.3f} (p = {p_rad:.2e})\n\n"
        f"Radiation Only (N = 17):\n"
        f"  Pearson r = {r_ro:+.3f} (p = {p_ro:.3f})"
    )
    ax.text(0.04, 0.95, stat_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    
    slc_stats.append({
        'Gene': g,
        'Name': full_name,
        'Pearson_r_All': r_val, 'Pearson_p_All': p_val,
        'Spearman_rho_All': rho_val, 'Spearman_p_All': sp_val,
        'Pearson_r_AllRad': r_rad, 'Pearson_p_AllRad': p_rad,
        'Pearson_r_RadOnly': r_ro, 'Pearson_p_RadOnly': p_ro
    })

plt.suptitle("SLC38 Family Neutral Amino Acid Transporters vs. Radiosensitivity Index (RSI) in TCGA-GBM",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()

f19_png = os.path.join(plots_dir, "19_RSI_vs_SLC38A_family_scatters.png")
f19_pdf = os.path.join(plots_dir, "19_RSI_vs_SLC38A_family_scatters.pdf")
plt.savefig(f19_png, dpi=300)
plt.savefig(f19_pdf)
plt.close()

# ==============================================================================
# FIGURE 20: COMPLETE 11-GENE (+ 2 FOLATE + COMPOSITE SCORES) LOLLIPOP CHART
# ==============================================================================
print("Generating Figure 20: Complete 11-Gene Lollipop Chart...")
all_serine_features = [
    'PHGDH', 'PSAT1', 'PSPH', 'SHMT1', 'SHMT2', 'SLC1A4', 'SLC1A5', 'SLC12A4',
    'SLC38A1', 'SLC38A2', 'SLC38A5', 'MTHFD1', 'MTHFD2', 'Serine_Score_8', 'Serine_Score_11'
]

full_gene_corrs = []
for feat in all_serine_features:
    r_all, p_all = stats.pearsonr(df_merged['RSI'], df_merged[feat])
    rho_all, sp_all = stats.spearmanr(df_merged['RSI'], df_merged[feat])
    
    r_rad, p_rad = stats.pearsonr(df_rad_only['RSI'], df_rad_only[feat])
    rho_rad, sp_rad = stats.spearmanr(df_rad_only['RSI'], df_rad_only[feat])
    
    r_allrad, p_allrad = stats.pearsonr(df_all_rad['RSI'], df_all_rad[feat])
    rho_allrad, sp_allrad = stats.spearmanr(df_all_rad['RSI'], df_all_rad[feat])
    
    category = "Score" if "Score" in feat else ("SLC38 Family" if "SLC38" in feat else ("Core Serine" if feat in ['PHGDH', 'PSAT1', 'PSPH', 'SHMT1', 'SHMT2', 'SLC1A4', 'SLC1A5', 'SLC12A4'] else "One-Carbon Folate"))
    
    full_gene_corrs.append({
        'Feature': feat,
        'Category': category,
        'Pearson_r_All': r_all, 'Pearson_p_All': p_all, 'Spearman_rho_All': rho_all, 'Spearman_p_All': sp_all,
        'Pearson_r_RadOnly': r_rad, 'Pearson_p_RadOnly': p_rad, 'Spearman_rho_RadOnly': rho_rad, 'Spearman_p_RadOnly': sp_rad,
        'Pearson_r_AllRad': r_allrad, 'Pearson_p_AllRad': p_allrad, 'Spearman_rho_AllRad': rho_allrad, 'Spearman_p_AllRad': sp_allrad
    })

df_full_corrs = pd.DataFrame(full_gene_corrs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8.5), sharey=True)

df_pop = df_full_corrs.sort_values('Spearman_rho_All', ascending=True).copy()
y_pos = np.arange(len(df_pop))

# Panel 1: All Patients
colors_all = ['#2b7bba' if r < 0 else '#d95f02' for r in df_pop['Spearman_rho_All']]
ax1.hlines(y=y_pos, xmin=0, xmax=df_pop['Spearman_rho_All'], color=colors_all, alpha=0.7, linewidth=2.5)
ax1.scatter(df_pop['Spearman_rho_All'], y_pos, color=colors_all, s=140, edgecolors='black', linewidth=1.2, zorder=3)

for idx, (rho, p, feat) in enumerate(zip(df_pop['Spearman_rho_All'], df_pop['Spearman_p_All'], df_pop['Feature'])):
    sig = "****" if p < 0.0001 else ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")))
    offset = 0.02 if rho >= 0 else -0.02
    ha = 'left' if rho >= 0 else 'right'
    weight_text = "bold" if sig != "ns" else "normal"
    ax1.text(rho + offset, idx, f"{rho:+.3f} ({sig})", va='center', ha=ha, fontsize=9.5, fontweight=weight_text)

ax1.axvline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_pop['Feature'], fontsize=11, fontweight='bold')
ax1.set_xlim(-0.45, 0.45)
ax1.set_xlabel("Spearman Correlation (\u03c1)", fontsize=11, fontweight='bold')
ax1.set_title("A. All TCGA-GBM Patients (N = 280)", fontsize=12, fontweight='bold')

# Panel 2: All Radiation-Treated Non-Immuno (N = 201)
colors_allrad = ['#2b7bba' if r < 0 else '#d95f02' for r in df_pop['Spearman_rho_AllRad']]
ax2.hlines(y=y_pos, xmin=0, xmax=df_pop['Spearman_rho_AllRad'], color=colors_allrad, alpha=0.7, linewidth=2.5)
ax2.scatter(df_pop['Spearman_rho_AllRad'], y_pos, color=colors_allrad, s=140, edgecolors='black', linewidth=1.2, zorder=3)

for idx, (rho, p) in enumerate(zip(df_pop['Spearman_rho_AllRad'], df_pop['Spearman_p_AllRad'])):
    sig = "****" if p < 0.0001 else ("***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns")))
    offset = 0.02 if rho >= 0 else -0.02
    ha = 'left' if rho >= 0 else 'right'
    weight_text = "bold" if sig != "ns" else "normal"
    ax2.text(rho + offset, idx, f"{rho:+.3f} ({sig})", va='center', ha=ha, fontsize=9.5, fontweight=weight_text)

ax2.axvline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlim(-0.45, 0.45)
ax2.set_xlabel("Spearman Correlation (\u03c1)", fontsize=11, fontweight='bold')
ax2.set_title("B. All Radiation-Treated Non-Immuno Cohort (N = 201)", fontsize=12, fontweight='bold')

plt.suptitle("Complete 11-Gene Serine/Amino-Acid Transport Signature vs. Radiosensitivity Index (RSI)",
             fontsize=13.5, fontweight='bold', y=0.99)
plt.tight_layout()

f20_png = os.path.join(plots_dir, "20_RSI_vs_complete_11gene_serine_lollipop.png")
f20_pdf = os.path.join(plots_dir, "20_RSI_vs_complete_11gene_serine_lollipop.pdf")
plt.savefig(f20_png, dpi=300)
plt.savefig(f20_pdf)
plt.close()

# ==============================================================================
# FIGURE 21: 11-GENE SERINE SCORE ACROSS TREATMENT COHORTS
# ==============================================================================
print("Generating Figure 21: 11-Gene Serine Score Treatment Panel...")
fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
axes = axes.flatten()

panels_11 = [
    ("A. Radiation Therapy Only (Monotherapy)", df_rad_only, '#e41a1c', axes[0]),
    ("B. All Radiation-Treated Non-Immuno", df_all_rad, '#2b7bba', axes[1]),
    ("C. Combination Chemoradiotherapy (Stupp)", df_combo, '#3182bd', axes[2]),
    ("D. Treatment Naive / Resection Only", df_naive, '#4daf4a', axes[3])
]

corr_11_rows = []

for title, sub, col, ax in panels_11:
    pr, pp = stats.pearsonr(sub['Serine_Score_11'], sub['RSI'])
    sr, sp = stats.spearmanr(sub['Serine_Score_11'], sub['RSI'])
    sl, ic, _, _, _ = stats.linregress(sub['Serine_Score_11'], sub['RSI'])
    
    ax.scatter(sub['Serine_Score_11'], sub['RSI'], color=col, s=70, alpha=0.75,
               edgecolors='black', linewidth=0.8, zorder=3)
    
    sns.regplot(data=sub, x='Serine_Score_11', y='RSI', scatter=False, ax=ax,
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
    
    corr_11_rows.append({
        'Cohort': title,
        'Sample_Size_N': len(sub),
        'Pearson_r': pr,
        'Pearson_p_value': pp,
        'Spearman_rho': sr,
        'Spearman_p_value': sp,
        'Slope': sl,
        'Intercept': ic
    })

plt.suptitle("Expanded 11-Gene Serine Score vs. RSI Across Clinical Treatment Cohorts",
             fontsize=13.5, fontweight='bold', y=0.995)
plt.tight_layout()

f21_png = os.path.join(plots_dir, "21_RSI_vs_Serine_Score_11_treatment_panel.png")
f21_pdf = os.path.join(plots_dir, "21_RSI_vs_Serine_Score_11_treatment_panel.pdf")
plt.savefig(f21_png, dpi=300)
plt.savefig(f21_pdf)
plt.close()

# ==============================================================================
# EXPORT STATISTICAL DATA TABLES
# ==============================================================================
print("Exporting updated correlation statistics tables...")
df_full_corrs.to_csv(os.path.join(tables_dir, "RSI_vs_complete_11gene_serine_correlations.csv"), index=False)
pd.DataFrame(slc_stats).to_csv(os.path.join(tables_dir, "RSI_vs_SLC38A_family_correlations.csv"), index=False)
pd.DataFrame(corr_11_rows).to_csv(os.path.join(tables_dir, "RSI_vs_Serine_Score_11_cohort_correlations.csv"), index=False)

# Update Excel workbook
excel_corr_path = os.path.join(tables_dir, "RSI_vs_Serine_Score_correlation_summary.xlsx")
with pd.ExcelWriter(excel_corr_path, engine='openpyxl') as writer:
    pd.read_csv(os.path.join(tables_dir, "RSI_vs_Serine_Score_cohort_correlations.csv")).to_excel(writer, sheet_name="8Gene_Cohort_Correlations", index=False)
    pd.DataFrame(corr_11_rows).to_excel(writer, sheet_name="11Gene_Cohort_Correlations", index=False)
    pd.DataFrame(slc_stats).to_excel(writer, sheet_name="SLC38A_Family_Correlations", index=False)
    df_full_corrs.to_excel(writer, sheet_name="Complete_11Gene_Lollipop", index=False)

print("All 11-gene analyses complete! Figures and tables saved.")
