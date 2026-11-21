import os
import requests
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Directory setup
workspace_dir = r"d:\DATA\TIGER_analysis\TIGER_analysis"
correlation_dir = os.path.join(workspace_dir, "correlation")

# All 8 Serine Metabolism Genes to analyze
target_genes = ["SLC1A5", "PHGDH", "PSPH", "PSAT1", "SLC1A4", "SLC12A4", "SHMT1", "SHMT2"]

# 12 Immune Markers
immune_markers = {
    "CD4": "CD4",
    "CD8A": "CD8A",
    "CD8B": "CD8B",
    "CD68": "CD68",
    "CD163": "CD163",
    "CD274": "PD-L1",
    "HLA-A": "HLA-A",
    "HLA-DRA": "HLA-DRA",
    "ITGAM": "CD11b",
    "PTPRC": "CD45",
    "B2M": "B2M",
    "FOXP3": "FOXP3"
}

# Fetch the TPM data (read from our local matched CSV to avoid repeated API requests)
tpm_csv_path = os.path.join(correlation_dir, "matched_expression_tpm.csv")
if not os.path.exists(tpm_csv_path):
    raise Exception("TPM csv data not found. Please run analyze_correlation.py first.")

df = pd.read_csv(tpm_csv_path)

# Create log2 version
all_columns = target_genes + list(immune_markers.values())
df_log2 = df.copy()
for col in all_columns:
    df_log2[col] = np.log2(df_log2[col] + 1)

# Loop through each target gene
for target in target_genes:
    # Create individual target directory
    target_dir = os.path.join(correlation_dir, target)
    os.makedirs(target_dir, exist_ok=True)
    print(f"\nGenerating individual TPM plots for {target} in: {target_dir}")
    
    # Generate scatter plot for each immune marker
    for cbio_symbol, marker in immune_markers.items():
        # Perform correlations
        r_val, p_val = stats.pearsonr(df_log2[target], df_log2[marker])
        rho_val, spearman_p = stats.spearmanr(df_log2[target], df_log2[marker])
        
        # 1. Generate Individual Square Scatter Plot (5x5 inches)
        plt.figure(figsize=(5, 5))
        
        # Plot regression
        sns.regplot(
            x=target, 
            y=marker, 
            data=df_log2,
            scatter_kws={"color": "#4a90e2", "alpha": 0.5, "s": 35, "edgecolors": "none"},
            line_kws={"color": "#d0021b", "linewidth": 2, "alpha": 0.8}
        )
        
        # Style titles and labels
        plt.title(f"{target} vs {marker} Co-expression", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel(f"{target} Expression [log2(TPM+1)]", fontsize=11, fontweight="bold")
        plt.ylabel(f"{marker} Expression [log2(TPM+1)]", fontsize=11, fontweight="bold")
        
        # Format p-values beautifully
        p_str = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.4f}"
        spearman_p_str = f"p = {spearman_p:.2e}" if spearman_p < 0.001 else f"p = {spearman_p:.4f}"
        
        stats_text = (
            f"Pearson r = {r_val:.3f}\n"
            f"  {p_str}\n"
            f"Spearman r\u03c5 = {rho_val:.3f}\n"
            f"  {spearman_p_str}"
        )
        
        # Place text box in the upper left with white semi-transparent background
        plt.gca().text(
            0.05, 0.95, stats_text, 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
        )
        
        # Save individual plot
        plot_filename = f"{target}_vs_{marker}_TPM.png"
        plot_path = os.path.join(target_dir, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Saved: {plot_filename}")

    # 2. Copy the combined grid and lollipop plot for this target into the folder for convenience
    combined_scatter_src = os.path.join(correlation_dir, f"{target}_GBM_correlation_scatters_TPM.png")
    combined_lollipop_src = os.path.join(correlation_dir, f"{target}_GBM_correlation_lollipop_TPM.png")
    
    combined_scatter_dst = os.path.join(target_dir, f"{target}_GBM_combined_scatters_grid_TPM.png")
    combined_lollipop_dst = os.path.join(target_dir, f"{target}_GBM_combined_lollipop_TPM.png")
    
    import shutil
    if os.path.exists(combined_scatter_src):
        shutil.copy(combined_scatter_src, combined_scatter_dst)
        print(f"  Copied combined scatter grid to: {os.path.basename(combined_scatter_dst)}")
    if os.path.exists(combined_lollipop_src):
        shutil.copy(combined_lollipop_src, combined_lollipop_dst)
        print(f"  Copied combined lollipop to: {os.path.basename(combined_lollipop_dst)}")

print("\nIndividual publication-quality TPM plots generated successfully for all 8 target genes.")
