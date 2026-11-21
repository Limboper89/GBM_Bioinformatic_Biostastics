import os
import gzip
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Set style for publication-quality plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Directory setup
workspace_dir = r"d:\DATA\TIGER_analysis\TIGER_analysis"
gtex_file = os.path.join(workspace_dir, "GTEx_data", "gene_tpm_v10_brain_cerebellar_hemisphere.gct.gz")
correlation_gtex_dir = os.path.join(workspace_dir, "correlation_gtex")
os.makedirs(correlation_gtex_dir, exist_ok=True)

# 8 Serine Metabolism Genes
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

all_genes_to_find = target_genes + list(immune_markers.keys())

# Benjamini-Hochberg FDR correction function
def benjamini_hochberg(p_values):
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    adjusted_p = np.zeros(n)
    prev_value = 1.0
    for idx in range(n - 1, -1, -1):
        p_val = sorted_p[idx]
        q_val = p_val * n / (idx + 1)
        q_val = min(q_val, prev_value)
        q_val = min(q_val, 1.0)
        adjusted_p[idx] = q_val
        prev_value = q_val
        
    original_order_adjusted_p = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        original_order_adjusted_p[idx] = adjusted_p[i]
        
    return list(original_order_adjusted_p)

# 1. Parse GTEx GCT.gz file
print("Step 1: Reading and parsing GTEx brain cerebellar hemisphere data...")
expression_data = {}
with gzip.open(gtex_file, "rt") as f:
    f.readline()  # skip version line
    f.readline()  # skip dimensions line
    headers = f.readline().strip().split("\t")
    sample_ids = headers[2:]
    
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) > 1:
            gene_symbol = parts[1]
            if gene_symbol in all_genes_to_find:
                vals = [float(val) for val in parts[2:]]
                expression_data[gene_symbol] = vals

# Create DataFrame
df = pd.DataFrame.from_dict(expression_data)
df.index = sample_ids
df.index.name = 'Sample_ID'
df = df.reset_index()

# Rename columns to display symbols
df = df.rename(columns=immune_markers)
all_columns = target_genes + list(immune_markers.values())

# Drop rows with NaNs in any gene
df = df.dropna(subset=all_columns)
N_samples = len(df)
print(f"Successfully loaded and aligned GTEx dataset. N = {N_samples} samples.")

# Save the matched dataset to CSV for reference
df.to_csv(os.path.join(correlation_gtex_dir, "matched_expression_tpm.csv"), index=False)

# Create log2 version
df_log2 = df.copy()
for col in all_columns:
    df_log2[col] = np.log2(df_log2[col] + 1)

# Store stats dataframes for the master Excel writer
master_stats = {}

# 2. Run analysis for each target gene
for target in target_genes:
    print(f"\nAnalyzing target gene: {target}...")
    
    target_dir = os.path.join(correlation_gtex_dir, target)
    os.makedirs(target_dir, exist_ok=True)
    
    excel_path = os.path.join(correlation_gtex_dir, f"{target}_GTEx_immune_correlation_TPM.xlsx")
    lollipop_path = os.path.join(correlation_gtex_dir, f"{target}_GTEx_correlation_lollipop_TPM.png")
    scatters_path = os.path.join(correlation_gtex_dir, f"{target}_GTEx_correlation_scatters_TPM.png")
    
    correlation_results = []
    markers = list(immune_markers.values())
    
    # 2.1 Calculate correlation coefficients and p-values
    for marker in markers:
        pearson_r_raw, pearson_p_raw = stats.pearsonr(df[target], df[marker])
        pearson_r_log2, pearson_p_log2 = stats.pearsonr(df_log2[target], df_log2[marker])
        spearman_rho, spearman_p = stats.spearmanr(df[target], df[marker])
        
        correlation_results.append({
            "Immune_Marker": marker,
            "Pearson_r_raw": pearson_r_raw,
            "Pearson_p_raw": pearson_p_raw,
            "Pearson_raw_FDR_q": 0.0,
            "Pearson_r_log2": pearson_r_log2,
            "Pearson_p_log2": pearson_p_log2,
            "Pearson_log2_FDR_q": 0.0,
            "Spearman_rho": spearman_rho,
            "Spearman_p_value": spearman_p,
            "Spearman_FDR_q": 0.0
        })
        
    df_stats = pd.DataFrame(correlation_results)
    
    # Apply Benjamini-Hochberg FDR correction
    df_stats["Pearson_raw_FDR_q"] = benjamini_hochberg(df_stats["Pearson_p_raw"])
    df_stats["Pearson_log2_FDR_q"] = benjamini_hochberg(df_stats["Pearson_p_log2"])
    df_stats["Spearman_FDR_q"] = benjamini_hochberg(df_stats["Spearman_p_value"])
    
    # Reorder columns
    column_order = [
        "Immune_Marker", 
        "Pearson_r_raw", "Pearson_p_raw", "Pearson_raw_FDR_q",
        "Pearson_r_log2", "Pearson_p_log2", "Pearson_log2_FDR_q",
        "Spearman_rho", "Spearman_p_value", "Spearman_FDR_q"
    ]
    df_stats = df_stats[column_order]
    master_stats[target] = df_stats
    
    # Save to individual Excel sheet in parent directory
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_stats.to_excel(writer, sheet_name="Correlation_Overview", index=False)
        for marker in markers:
            marker_df = pd.DataFrame({
                "Sample_ID": df["Sample_ID"],
                f"{target}_TPM": df[target],
                f"{marker}_TPM": df[marker],
                "Spacer": [""] * len(df),
                f"{target}_Log2_TPM": df_log2[target],
                f"{marker}_Log2_TPM": df_log2[marker]
            })
            marker_df.to_excel(writer, sheet_name=marker, index=False)
            
    # 2.2 Generate lollipop plot
    df_lollipop = df_stats.sort_values(by="Spearman_rho", ascending=True)
    sig_labels = []
    for p in df_lollipop["Spearman_p_value"]:
        if p < 0.0001:
            sig_labels.append("****")
        elif p < 0.001:
            sig_labels.append("***")
        elif p < 0.01:
            sig_labels.append("**")
        elif p < 0.05:
            sig_labels.append("*")
        else:
            sig_labels.append("ns")
    df_lollipop["significance"] = sig_labels
    
    plt.figure(figsize=(10, 6))
    colors = ["#2b7bba" if r < 0 else "#d95f02" for r in df_lollipop["Spearman_rho"]]
    y_pos = np.arange(len(df_lollipop))
    plt.hlines(y=y_pos, xmin=0, xmax=df_lollipop["Spearman_rho"], color=colors, alpha=0.6, linewidth=2)
    plt.scatter(df_lollipop["Spearman_rho"], y_pos, color=colors, s=120, alpha=0.9, edgecolors="black", zorder=3)
    
    for idx, (rho, sig) in enumerate(zip(df_lollipop["Spearman_rho"], df_lollipop["significance"])):
        offset = 0.02 if rho >= 0 else -0.02
        ha = "left" if rho >= 0 else "right"
        plt.text(rho + offset, idx, f"{rho:.3f} ({sig})", va="center", ha=ha, fontsize=10, fontweight="bold")
        
    plt.yticks(y_pos, df_lollipop["Immune_Marker"], fontsize=11, fontweight="bold")
    plt.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    
    min_x = min(0.0, df_lollipop["Spearman_rho"].min()) - 0.15
    max_x = max(0.0, df_lollipop["Spearman_rho"].max()) + 0.15
    plt.xlim(min_x, max_x)
    plt.xlabel("Spearman Correlation Coefficient (r\u03c5)", fontsize=12, fontweight="bold")
    plt.title(f"Correlation of {target} with Immune Markers in GTEx Normal Brain\nSorted by Spearman (TPM values)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(lollipop_path, dpi=300)
    plt.close()
    
    # 2.3 Generate scatter plot grid (4x3)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, marker in enumerate(markers):
        ax = axes[idx]
        marker_stats = df_stats[df_stats["Immune_Marker"] == marker].iloc[0]
        r_val = marker_stats["Pearson_r_log2"]
        p_val = marker_stats["Pearson_p_log2"]
        rho_val = marker_stats["Spearman_rho"]
        spearman_p = marker_stats["Spearman_p_value"]
        
        sns.regplot(
            x=target, 
            y=marker, 
            data=df_log2, 
            ax=ax,
            scatter_kws={"color": "#4a90e2", "alpha": 0.5, "s": 25, "edgecolors": "none"},
            line_kws={"color": "#d0021b", "linewidth": 2, "alpha": 0.8}
        )
        
        ax.set_title(f"{target} vs {marker}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(f"{target} [TPM log2]")
        ax.set_ylabel(f"{marker} [TPM log2]")
        
        p_str = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.4f}"
        spearman_p_str = f"p = {spearman_p:.2e}" if spearman_p < 0.001 else f"p = {spearman_p:.4f}"
        
        stats_text = (
            f"Pearson r = {r_val:.3f}\n"
            f"  {p_str}\n"
            f"Spearman r\u03c5 = {rho_val:.3f}\n"
            f"  {spearman_p_str}"
        )
        ax.text(
            0.05, 0.95, stats_text, 
            transform=ax.transAxes, 
            fontsize=10.5, 
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
        )
        
    plt.suptitle(f"Co-expression Profiling of {target} and Immune Signatures in GTEx Normal Brain (N=277, TPM)", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(scatters_path, dpi=300)
    plt.close()
    
    # 2.4 Generate 12 individual publication-quality square scatter plots (5x5 inches)
    print(f"  Generating 12 individual scatter plots for {target}...")
    for marker in markers:
        plt.figure(figsize=(5, 5))
        
        sns.regplot(
            x=target, 
            y=marker, 
            data=df_log2,
            scatter_kws={"color": "#4a90e2", "alpha": 0.5, "s": 35, "edgecolors": "none"},
            line_kws={"color": "#d0021b", "linewidth": 2, "alpha": 0.8}
        )
        
        plt.title(f"{target} vs {marker} Co-expression (GTEx)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel(f"{target} Expression [log2(TPM+1)]", fontsize=11, fontweight="bold")
        plt.ylabel(f"{marker} Expression [log2(TPM+1)]", fontsize=11, fontweight="bold")
        
        marker_stats = df_stats[df_stats["Immune_Marker"] == marker].iloc[0]
        r_val = marker_stats["Pearson_r_log2"]
        p_val = marker_stats["Pearson_p_log2"]
        rho_val = marker_stats["Spearman_rho"]
        spearman_p = marker_stats["Spearman_p_value"]
        
        p_str = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.4f}"
        spearman_p_str = f"p = {spearman_p:.2e}" if spearman_p < 0.001 else f"p = {spearman_p:.4f}"
        
        stats_text = (
            f"Pearson r = {r_val:.3f}\n"
            f"  {p_str}\n"
            f"Spearman r\u03c5 = {rho_val:.3f}\n"
            f"  {spearman_p_str}"
        )
        
        plt.gca().text(
            0.05, 0.95, stats_text, 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5)
        )
        
        plot_path = os.path.join(target_dir, f"{target}_vs_{marker}_TPM.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
    # Copy statistical analysis Excel and combined figures into folder
    shutil.copy(excel_path, os.path.join(target_dir, f"{target}_GTEx_immune_correlation_TPM.xlsx"))
    shutil.copy(lollipop_path, os.path.join(target_dir, f"{target}_GTEx_combined_lollipop_TPM.png"))
    shutil.copy(scatters_path, os.path.join(target_dir, f"{target}_GTEx_combined_scatters_grid_TPM.png"))
    print(f"  Saved all individual outputs in: {target_dir}")

# 3. Save the Master Excel file for stats
master_excel_path = os.path.join(correlation_gtex_dir, "Serine_Metabolism_GTEx_Immune_Correlation_Master_TPM.xlsx")
with pd.ExcelWriter(master_excel_path, engine='openpyxl') as master_writer:
    for target in target_genes:
        master_stats[target].to_excel(master_writer, sheet_name=target, index=False)
print(f"\nSaved master correlation statistical sheet to: {master_excel_path}")

# 4. Generate Prism-friendly Raw and Log2 spreadsheets
raw_prism_path = os.path.join(correlation_gtex_dir, "Serine_Metabolism_GTEx_Prism_Raw_TPM.xlsx")
log2_prism_path = os.path.join(correlation_gtex_dir, "Serine_Metabolism_GTEx_Prism_Log2_TPM.xlsx")

# 4.1 Raw values Prism Excel
with pd.ExcelWriter(raw_prism_path, engine='openpyxl') as writer_raw:
    for target in target_genes:
        raw_columns = ["Sample_ID", target] + list(immune_markers.values())
        df_raw_subset = df[raw_columns].copy()
        df_raw_subset.to_excel(writer_raw, sheet_name=target, index=False)

# 4.2 Log2 values Prism Excel
with pd.ExcelWriter(log2_prism_path, engine='openpyxl') as writer_log2:
    for target in target_genes:
        raw_columns = ["Sample_ID", target] + list(immune_markers.values())
        df_log2_subset = df_log2[raw_columns].copy()
        # Rename columns to show _Log2
        rename_dict_log2 = {target: f"{target}_Log2"}
        for m in immune_markers.values():
            rename_dict_log2[m] = f"{m}_Log2"
        df_log2_subset = df_log2_subset.rename(columns=rename_dict_log2)
        df_log2_subset.to_excel(writer_log2, sheet_name=target, index=False)

print(f"Saved Prism-friendly Raw values master: {raw_prism_path}")
print(f"Saved Prism-friendly Log2 values master: {log2_prism_path}")

print("\nGTEx normal tissue correlation analysis complete. All outputs successfully saved in correlation_gtex/.")
