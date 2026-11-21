import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# Set plotting style for premium look
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

# 1. Locate and Load Data
clinical_file = 'GBM-PRJNA482620.Response_clinical.tsv.xlsx'
expression_file = 'GBM-PRJNA482620.Response.tsv'

# Check in current directory or parent directory
if not os.path.exists(clinical_file):
    clinical_file = os.path.join('..', clinical_file)
if not os.path.exists(expression_file):
    expression_file = os.path.join('..', expression_file)

if not os.path.exists(clinical_file) or not os.path.exists(expression_file):
    raise FileNotFoundError("Could not find the clinical or expression data files in current or parent directory.")

print("Loading clinical data from:", clinical_file)
df_clin = pd.read_excel(clinical_file)

print("Loading expression data from:", expression_file)
df_expr = pd.read_csv(expression_file, sep='\t')

# Target Genes (corrected typo MTFHD1 -> MTHFD1 and deduplicated SLC1A4, added SLC12A4, SLC1A5)
target_genes = ['PHGDH', 'PSPH', 'PSAT1', 'SHMT1', 'SHMT2', 'SLC1A4', 'MTHFD1', 'MTHFD2', 'SLC12A4', 'SLC1A5']

# 2. Extract and Filter Gene Expression
print("Filtering expression data for target genes...")
df_expr_target = df_expr[df_expr['GENE_SYMBOL'].isin(target_genes)].set_index('GENE_SYMBOL')

# Transpose expression data to merge with clinical metadata
expr_df = df_expr_target.T

# Merge with clinical data to map sample names to patient IDs
merged_df = expr_df.merge(
    df_clin[['patient_name', 'dataset_id', 'overall survival (days)', 'vital status', 'response_NR']],
    left_index=True,
    right_on='patient_name'
)

# 3. Replicate TIGER Webserver Cohort (N = 27 samples)
# Exclude the 7 post-treatment / replicate runs determined by search
excluded_runs = ['SRR8281218', 'SRR8281219', 'SRR8281221', 'SRR8281222', 'SRR8281224', 'SRR8281227', 'SRR8281237']
print(f"Replicating TIGER webserver analysis. Excluding {len(excluded_runs)} runs...")
tiger_df = merged_df[~merged_df['patient_name'].isin(excluded_runs)].copy()

print(f"Total samples in replication cohort: {len(tiger_df)}")

# Map vital status to numeric event (1 = Dead, 0 = Alive/Censored)
tiger_df['Status'] = (tiger_df['vital status'] == 'Dead').astype(int)
tiger_df['Time'] = tiger_df['overall survival (days)']

# Save Master Dataset in the current folder (tiger_webserver_replication)
print("Saving TIGER replication master dataset...")
tiger_df.to_csv('gbm_anti_pd1_tiger_master.csv', index=False)

# 4. Generate Kaplan-Meier Curves and Prism-friendly CSV Structure
prism_cols = {}

# Set colors for plotting
high_color = '#E05D5D'  # Premium coral-red
low_color = '#007A87'   # Premium teal-blue

print("\n--- Running Log-Rank Tests and Generating Files ---")

# We will save single plots and compile a 3x4 grid plot
fig_grid, axes_grid = plt.subplots(3, 4, figsize=(20, 15))
axes_flat = axes_grid.flatten()

for idx, gene in enumerate(target_genes):
    # Calculate median for the gene across the 27 samples
    median_val = tiger_df[gene].median()
    
    # Classify samples into High (> median) and Low (<= median) groups
    tiger_df[f'{gene}_Group'] = np.where(tiger_df[gene] > median_val, 'High', 'Low')
    
    # Split patient data
    high_group = tiger_df[tiger_df[f'{gene}_Group'] == 'High']
    low_group = tiger_df[tiger_df[f'{gene}_Group'] == 'Low']
    
    # Log-rank test
    res = logrank_test(
        high_group['Time'], low_group['Time'],
        event_observed_A=high_group['Status'], event_observed_B=low_group['Status']
    )
    p_val = res.p_value
    p_str = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
    sig_marker = " *" if p_val < 0.05 else ""
    print(f"Gene: {gene:7s} | Median: {median_val:8.4f} | High N: {len(high_group)} | Low N: {len(low_group)} | Log-rank p-value: {p_val:.5f}{sig_marker}")
    
    # Create Prism-formatted columns for this gene
    # Sort samples by time for a clean Prism table appearance
    gene_df = tiger_df[['Time', 'Status', f'{gene}_Group']].copy()
    gene_df = gene_df.sort_values(by='Time')
    
    time_col = f'{gene}_Time'
    high_col = f'{gene}_High'
    low_col = f'{gene}_Low'
    
    gene_prism = pd.DataFrame(index=gene_df.index)
    gene_prism[time_col] = gene_df['Time']
    gene_prism[high_col] = np.nan
    gene_prism[low_col] = np.nan
    
    gene_prism.loc[gene_df[f'{gene}_Group'] == 'High', high_col] = gene_df['Status']
    gene_prism.loc[gene_df[f'{gene}_Group'] == 'Low', low_col] = gene_df['Status']
    
    gene_prism = gene_prism.reset_index(drop=True)
    
    prism_cols[time_col] = gene_prism[time_col]
    prism_cols[high_col] = gene_prism[high_col]
    prism_cols[low_col] = gene_prism[low_col]
    prism_cols[f'{gene}_Spacer'] = '' # empty separator column
    
    # --- Individual Kaplan-Meier Plotting ---
    fig_ind, ax_ind = plt.subplots(figsize=(7, 6))
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    # Fit data
    kmf_high.fit(high_group['Time'], high_group['Status'], label=f'High {gene} (N={len(high_group)})')
    kmf_low.fit(low_group['Time'], low_group['Status'], label=f'Low {gene} (N={len(low_group)})')
    
    # Plot curves
    for ax_temp, is_grid in [(ax_ind, False), (axes_flat[idx], True)]:
        kmf_high.plot_survival_function(
            ax=ax_temp, color=high_color, lw=2.5, ci_show=False,
            show_censors=True, censor_styles={'ms': 8, 'mew': 1.5, 'marker': '+'}
        )
        kmf_low.plot_survival_function(
            ax=ax_temp, color=low_color, lw=2.5, ci_show=False,
            show_censors=True, censor_styles={'ms': 8, 'mew': 1.5, 'marker': '+'}
        )
        
        # Style adjustments
        ax_temp.set_ylim(0, 1.05)
        ax_temp.set_xlim(0, 1800)
        ax_temp.set_ylabel("Survival Probability", fontsize=11, fontweight='medium')
        ax_temp.set_xlabel("Overall Survival (Days)", fontsize=11, fontweight='medium')
        
        # Display p-value
        ax_temp.text(
            0.05, 0.1, f"Log-rank\n{p_str}", transform=ax_temp.transAxes,
            fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#E2E8F0', alpha=0.9)
        )
        
        if is_grid:
            ax_temp.set_title(f"{gene} (Median split)", fontsize=13, fontweight='bold')
            ax_temp.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)
        else:
            ax_temp.set_title(f"Kaplan-Meier Survival Analysis for {gene} (TIGER Replication)", fontsize=14, fontweight='bold', pad=15)
            ax_temp.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=10)
            sns.despine(ax=ax_temp, offset=10, trim=True)
            
    fig_ind.tight_layout()
    add_at_risk_counts(kmf_high, kmf_low, ax=ax_ind, rows_to_show=['At risk'])
    
    # Save individual plots
    fig_ind.savefig(f"KM_curve_{gene}.png", bbox_inches='tight', dpi=300)
    fig_ind.savefig(f"KM_curve_{gene}.pdf", bbox_inches='tight')
    plt.close(fig_ind)

# Hide unused subplots in the grid
for idx_extra in range(len(target_genes), len(axes_flat)):
    axes_flat[idx_extra].axis('off')

# Configure and save the combined grid plot
fig_grid.suptitle("Kaplan-Meier Curves for Amino Acid & Folate Metabolism Genes in Glioblastoma\n(TIGER Replication Cohort, Sample-level Median Split, N=27 samples)", fontsize=18, fontweight='bold', y=0.98)
fig_grid.tight_layout(rect=[0, 0, 1, 0.95])
fig_grid.savefig("KM_curves_combined.png", bbox_inches='tight', dpi=300)
plt.close(fig_grid)

# 5. Save Prism-friendly CSV
print("\nSaving Prism-friendly CSV...")
prism_df = pd.DataFrame(prism_cols)
if f'{target_genes[-1]}_Spacer' in prism_df.columns:
    prism_df = prism_df.drop(columns=[f'{target_genes[-1]}_Spacer'])

prism_df.to_csv('gbm_anti_pd1_survival_prism.csv', index=False)
print("Prism-friendly file saved as 'gbm_anti_pd1_survival_prism.csv'")

print("Analysis run complete. All TIGER replication results saved successfully.")
