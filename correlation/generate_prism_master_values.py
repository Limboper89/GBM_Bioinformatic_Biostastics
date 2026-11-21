import os
import pandas as pd
import numpy as np

workspace_dir = r"d:\DATA\TIGER_analysis\TIGER_analysis"
correlation_dir = os.path.join(workspace_dir, "correlation")

target_genes = ["SLC1A5", "PHGDH", "PSPH", "PSAT1", "SLC1A4", "SLC12A4", "SHMT1", "SHMT2"]
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

for norm_type in ["tpm", "fpkm"]:
    csv_path = os.path.join(correlation_dir, f"matched_expression_{norm_type}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Skip.")
        continue
        
    df = pd.read_csv(csv_path)
    
    # Create log2 version
    all_cols = target_genes + list(immune_markers.values())
    df_log2 = df.copy()
    for col in all_cols:
        df_log2[col] = np.log2(df_log2[col] + 1)
        
    # File paths for the two clean sheets (Raw and Log2) for this norm_type
    raw_master_path = os.path.join(correlation_dir, f"Serine_Metabolism_GBM_Prism_Raw_{norm_type.upper()}.xlsx")
    log2_master_path = os.path.join(correlation_dir, f"Serine_Metabolism_GBM_Prism_Log2_{norm_type.upper()}.xlsx")
    
    # Generate Raw workbook (8 tabs: SLC1A5, PHGDH, PSPH, PSAT1, SLC1A4, SLC12A4, SHMT1, SHMT2)
    # Inside each tab, column A is Sample_ID, column B is Gene_X, and columns C-N are the 12 immune markers (Y)
    with pd.ExcelWriter(raw_master_path, engine='openpyxl') as writer_raw:
        for target in target_genes:
            raw_columns = ["Sample_ID", target] + list(immune_markers.values())
            df_raw_subset = df[raw_columns].copy()
            df_raw_subset.to_excel(writer_raw, sheet_name=target, index=False)
            
    # Generate Log2 workbook (8 tabs)
    # Inside each tab, column A is Sample_ID, column B is Gene_X_Log2, and columns C-N are the 12 immune markers (Y_Log2)
    with pd.ExcelWriter(log2_master_path, engine='openpyxl') as writer_log2:
        for target in target_genes:
            raw_columns = ["Sample_ID", target] + list(immune_markers.values())
            df_log2_subset = df_log2[raw_columns].copy()
            # Rename for clarity
            rename_dict_log2 = {target: f"{target}_Log2"}
            for m in immune_markers.values():
                rename_dict_log2[m] = f"{m}_Log2"
            df_log2_subset = df_log2_subset.rename(columns=rename_dict_log2)
            df_log2_subset.to_excel(writer_log2, sheet_name=target, index=False)
            
    print(f"Successfully generated clean Prism files for {norm_type.upper()}:")
    print(f"  - Raw values: {raw_master_path}")
    print(f"  - Log2 values: {log2_master_path}")
