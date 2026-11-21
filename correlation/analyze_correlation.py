import os
import requests
import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Directory setup
workspace_dir = r"d:\DATA\TIGER_analysis\TIGER_analysis"
correlation_dir = os.path.join(workspace_dir, "correlation")
os.makedirs(correlation_dir, exist_ok=True)

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

# Combine all genes for cBioPortal API queries
all_genes_to_query = target_genes + list(immune_markers.keys())

base_url = "https://www.cbioportal.org/api"
study_id = "gbm_tcga_gdc"
sample_list_id = "gbm_tcga_gdc_all"

# Define the two molecular profiles to analyze
profiles = {
    "TPM": "gbm_tcga_gdc_mrna_seq_tpm",
    "FPKM": "gbm_tcga_gdc_mrna_seq_fpkm"
}

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

# Fetch Entrez IDs
print("Step 1: Mapping gene symbols to Entrez IDs via cBioPortal API...")
entrez_ids = {}
for cbio_symbol in all_genes_to_query:
    url = f"{base_url}/genes/{cbio_symbol}"
    r = requests.get(url)
    if r.status_code == 200:
        entrez_ids[cbio_symbol] = r.json().get("entrezGeneId")
    else:
        raise Exception(f"Failed to map gene {cbio_symbol}")

# Loop through each profile (TPM and FPKM)
for norm_type, profile_id in profiles.items():
    print(f"\n=======================================================")
    print(f"PROCESS NORMALIZATION: {norm_type} ({profile_id})")
    print(f"=======================================================")
    
    # Fetch expression data
    print("Fetching expression data for all genes...")
    all_gene_data = {}
    for cbio_symbol, entrez_id in entrez_ids.items():
        url = f"{base_url}/molecular-profiles/{profile_id}/molecular-data?sampleListId={sample_list_id}&entrezGeneId={entrez_id}"
        r = requests.get(url)
        if r.status_code == 200:
            all_gene_data[cbio_symbol] = r.json()
        else:
            raise Exception(f"Failed to fetch molecular data for {cbio_symbol}")
            
    # Align expression data across all genes per sample
    aligned_data = {}
    for cbio_symbol, data in all_gene_data.items():
        if cbio_symbol in target_genes:
            user_symbol = cbio_symbol
        else:
            user_symbol = immune_markers[cbio_symbol]
            
        for record in data:
            sample_id = record.get("sampleId")
            val = record.get("value")
            if sample_id and val is not None:
                if sample_id not in aligned_data:
                    aligned_data[sample_id] = {}
                aligned_data[sample_id][user_symbol] = val
                
    # Create DataFrame
    df = pd.DataFrame.from_dict(aligned_data, orient='index')
    df.index.name = 'Sample_ID'
    df = df.reset_index()
    
    # Drop rows with NaNs in any target gene or immune marker
    all_columns = target_genes + list(immune_markers.values())
    df = df.dropna(subset=all_columns)
    
    # Create a log2(val + 1) version
    df_log2 = df.copy()
    for col in all_columns:
        df_log2[col] = np.log2(df_log2[col] + 1)
        
    print(f"Matched N={len(df)} samples.")
    
    # Save the complete aligned dataset for reference
    df.to_csv(os.path.join(correlation_dir, f"matched_expression_{norm_type.lower()}.csv"), index=False)
    
    # Store stats dataframes for the master Excel writer
    master_stats = {}
    
    # Run analysis for each target gene
    for target in target_genes:
        print(f"  Analyzing Target Gene: {target}...")
        
        excel_path = os.path.join(correlation_dir, f"{target}_GBM_immune_correlation_{norm_type}.xlsx")
        lollipop_path = os.path.join(correlation_dir, f"{target}_GBM_correlation_lollipop_{norm_type}.png")
        scatters_path = os.path.join(correlation_dir, f"{target}_GBM_correlation_scatters_{norm_type}.png")
        
        correlation_results = []
        markers = list(immune_markers.values())
        
        for marker in markers:
            # Pearson on raw values
            pearson_r_raw, pearson_p_raw = stats.pearsonr(df[target], df[marker])
            # Pearson on log2 values
            pearson_r_log2, pearson_p_log2 = stats.pearsonr(df_log2[target], df_log2[marker])
            # Spearman correlation (rank-based)
            spearman_rho, spearman_p = stats.spearmanr(df[target], df[marker])
            
            correlation_results.append({
                "Immune_Marker": marker,
                "Pearson_r_raw": pearson_r_raw,
                "Pearson_p_raw": pearson_p_raw,
                "Pearson_raw_FDR_q": 0.0,  # Will update after BH
                "Pearson_r_log2": pearson_r_log2,
                "Pearson_p_log2": pearson_p_log2,
                "Pearson_log2_FDR_q": 0.0,  # Will update after BH
                "Spearman_rho": spearman_rho,
                "Spearman_p_value": spearman_p,
                "Spearman_FDR_q": 0.0      # Will update after BH
            })
            
        df_stats = pd.DataFrame(correlation_results)
        
        # Apply BH correction
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
        
        # Save to our master stats dictionary
        master_stats[target] = df_stats
        
        # Write Excel for individual gene
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_stats.to_excel(writer, sheet_name="Correlation_Overview", index=False)
                for marker in markers:
                    marker_df = pd.DataFrame({
                        "Sample_ID": df["Sample_ID"],
                        f"{target}_{norm_type}": df[target],
                        f"{marker}_{norm_type}": df[marker],
                        "Spacer": [""] * len(df),
                        f"{target}_Log2_{norm_type}": df_log2[target],
                        f"{marker}_Log2_{norm_type}": df_log2[marker]
                    })
                    marker_df.to_excel(writer, sheet_name=marker, index=False)
        except PermissionError:
            fallback = excel_path.replace(".xlsx", "_LOCKED_FALLBACK.xlsx")
            print(f"    [WARNING] Permission denied. Saving to fallback: {fallback}")
            with pd.ExcelWriter(fallback, engine='openpyxl') as writer:
                df_stats.to_excel(writer, sheet_name="Correlation_Overview", index=False)
                for marker in markers:
                    marker_df = pd.DataFrame({
                        "Sample_ID": df["Sample_ID"],
                        f"{target}_{norm_type}": df[target],
                        f"{marker}_{norm_type}": df[marker],
                        "Spacer": [""] * len(df),
                        f"{target}_Log2_{norm_type}": df_log2[target],
                        f"{marker}_Log2_{norm_type}": df_log2[marker]
                    })
                    marker_df.to_excel(writer, sheet_name=marker, index=False)
                    
        # Generate lollipop plot
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
        plt.title(f"Correlation of {target} with Immune Markers in GBM (TCGA)\nSorted by Spearman ({norm_type} values)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(lollipop_path, dpi=300)
        plt.close()
        
        # Generate scatter plot grid (4x3)
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
                scatter_kws={"color": "#4a90e2", "alpha": 0.4, "s": 25, "edgecolors": "none"},
                line_kws={"color": "#d0021b", "linewidth": 2, "alpha": 0.8}
            )
            
            ax.set_title(f"{target} vs {marker}", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel(f"{target} [{norm_type} log2]")
            ax.set_ylabel(f"{marker} [{norm_type} log2]")
            
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
            
        plt.suptitle(f"Co-expression Profiling of {target} and Immune Signatures in Glioblastoma (TCGA, N=295, {norm_type})", fontsize=18, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(scatters_path, dpi=300)
        plt.close()
        
    # Write the Master Excel file for this normalization type
    master_excel_path = os.path.join(correlation_dir, f"Serine_Metabolism_GBM_Immune_Correlation_Master_{norm_type}.xlsx")
    try:
        with pd.ExcelWriter(master_excel_path, engine='openpyxl') as master_writer:
            for target in target_genes:
                master_stats[target].to_excel(master_writer, sheet_name=target, index=False)
        print(f"Master Excel file saved successfully: {master_excel_path}")
    except PermissionError:
        fallback_master = master_excel_path.replace(".xlsx", "_LOCKED_FALLBACK.xlsx")
        print(f"[WARNING] Master Excel locked. Saving to fallback: {fallback_master}")
        with pd.ExcelWriter(fallback_master, engine='openpyxl') as master_writer:
            for target in target_genes:
                master_stats[target].to_excel(master_writer, sheet_name=target, index=False)
        print(f"Master Excel saved to fallback: {fallback_master}")

print("\nBioinformatic analysis complete. Master Excel sheets and individual profiles generated successfully.")
