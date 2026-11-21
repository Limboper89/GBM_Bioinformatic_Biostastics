import os
import glob
import gzip
import requests
import json
import pandas as pd
import numpy as np

# Absolute paths
workspace_dir = r"d:\DATA\TIGER_analysis\TIGER_analysis"
gtex_file = os.path.join(workspace_dir, "GTEx_data", "gene_tpm_v10_brain_cerebellar_hemisphere.gct.gz")
output_file = os.path.join(workspace_dir, "slc1a5_expression_prism.csv")

print("Step 1: Scanning local TCGA directories for TSV files...")
gbm_dir = os.path.join(workspace_dir, "GDCdata", "TCGA-GBM")
lgg_dir = os.path.join(workspace_dir, "GDCdata", "TCGA-LGG")

gbm_files = glob.glob(os.path.join(gbm_dir, "**", "*.tsv"), recursive=True)
lgg_files = glob.glob(os.path.join(lgg_dir, "**", "*.tsv"), recursive=True)

print(f"Found {len(gbm_files)} TSV files in GDCdata/TCGA-GBM.")
print(f"Found {len(lgg_files)} TSV files in GDCdata/TCGA-LGG.")

tsv_files = gbm_files + lgg_files
print(f"Total TCGA TSV files to process: {len(tsv_files)}")

# Extract file UUIDs from path
file_info = []
for fpath in tsv_files:
    parts = fpath.split(os.sep)
    # The file UUID is the directory containing the file
    file_uuid = parts[-2]
    file_name = parts[-1]
    # Determine cohort based on path
    cohort = "GBM" if "TCGA-GBM" in fpath else "LGG"
    file_info.append({
        "file_uuid": file_uuid,
        "file_name": file_name,
        "path": fpath,
        "cohort": cohort
    })

print("\nStep 2: Mapping file UUIDs to TCGA sample barcodes using GDC API...")
# We will query GDC API in batches of 40 to avoid HTTP 414 Request-URI Too Large
uuid_to_meta = {}
batch_size = 40
file_uuids = [info["file_uuid"] for info in file_info]

for i in range(0, len(file_uuids), batch_size):
    batch = file_uuids[i:i+batch_size]
    print(f"Querying batch {i//batch_size + 1} ({len(batch)} UUIDs)...")
    
    url = "https://api.gdc.cancer.gov/files"
    params = {
        "filters": json.dumps({
            "op": "in",
            "content": {
                "field": "files.file_id",
                "value": batch
            }
        }),
        "fields": "file_id,file_name,cases.samples.submitter_id,cases.samples.sample_type",
        "format": "json",
        "size": str(len(batch))
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        hits = data.get("data", {}).get("hits", [])
        for hit in hits:
            fid = hit.get("file_id")
            cases = hit.get("cases", [])
            if cases:
                samples = cases[0].get("samples", [])
                if samples:
                    submitter_id = samples[0].get("submitter_id")
                    sample_type = samples[0].get("sample_type")
                    uuid_to_meta[fid] = {
                        "barcode": submitter_id,
                        "sample_type": sample_type
                    }
    else:
        print(f"Warning: Failed query for batch {i//batch_size + 1}. Status code {response.status_code}")

print(f"Successfully mapped {len(uuid_to_meta)} out of {len(file_uuids)} file UUIDs.")

print("\nStep 3: Extracting SLC1A5 expression values from TCGA TSV files...")
gbm_tumor_vals = []
lgg_tumor_vals = []
tcga_normal_vals = [] # Combine normal controls if any exist

for info in file_info:
    fid = info["file_uuid"]
    meta = uuid_to_meta.get(fid)
    if not meta:
        continue
    
    barcode = meta["barcode"]
    sample_type = meta["sample_type"]
    fpath = info["path"]
    cohort = info["cohort"]
    
    # Read the file and find SLC1A5
    tpm_val = None
    with open(fpath, "r") as f:
        # Skip header comment if it exists
        line = f.readline()
        if line.startswith("#"):
            headers = f.readline().strip().split("\t")
        else:
            headers = line.strip().split("\t")
            
        tpm_idx = headers.index("tpm_unstranded")
        name_idx = headers.index("gene_name")
        
        for row in f:
            parts = row.strip().split("\t")
            if len(parts) > max(name_idx, tpm_idx):
                if parts[name_idx] == "SLC1A5":
                    tpm_val = float(parts[tpm_idx])
                    break
                    
    if tpm_val is not None:
        if "Normal" in sample_type:
            tcga_normal_vals.append(tpm_val)
        elif "Tumor" in sample_type or "Metastatic" in sample_type or "Cancer" in sample_type:
            if cohort == "GBM":
                gbm_tumor_vals.append(tpm_val)
            else:
                lgg_tumor_vals.append(tpm_val)

print(f"Extracted {len(gbm_tumor_vals)} TCGA GBM Tumor values.")
print(f"Extracted {len(lgg_tumor_vals)} TCGA LGG Tumor values.")
print(f"Extracted {len(tcga_normal_vals)} TCGA Normal values.")

print("\nStep 4: Extracting SLC1A5 expression values from GTEx cerebellar hemisphere file...")
gtex_vals = []
if os.path.exists(gtex_file):
    with gzip.open(gtex_file, "rt") as f:
        # Skip metadata
        line1 = f.readline() # version
        line2 = f.readline() # dimensions
        headers = f.readline().strip().split("\t") # columns
        
        found = False
        for row in f:
            parts = row.strip().split("\t")
            if len(parts) > 1 and parts[1] == "SLC1A5":
                # The first two columns are Name and Description, followed by sample TPMs
                gtex_vals = [float(val) for val in parts[2:]]
                found = True
                break
        if found:
            print(f"Extracted {len(gtex_vals)} GTEx Normal values.")
        else:
            print("Warning: SLC1A5 not found in GTEx file!")
else:
    print(f"Error: GTEx file not found at {gtex_file}")

print("\nStep 5: Formatting and compiling data for GraphPad Prism...")
# Calculate log2(TPM + 1)
gtex_log2 = [np.log2(val + 1) for val in gtex_vals]
gbm_tumor_log2 = [np.log2(val + 1) for val in gbm_tumor_vals]
lgg_tumor_log2 = [np.log2(val + 1) for val in lgg_tumor_vals]

# Find max length to align columns (unequal rows)
max_len = max(len(gtex_vals), len(gbm_tumor_vals), len(lgg_tumor_vals))

def pad_list(lst, target_len):
    return lst + [np.nan] * (target_len - len(lst))

# Prepare columns
prism_data = {
    "GTEx_Normal_TPM": pad_list(gtex_vals, max_len),
    "TCGA_GBM_Tumor_TPM": pad_list(gbm_tumor_vals, max_len),
    "TCGA_LGG_Tumor_TPM": pad_list(lgg_tumor_vals, max_len),
    
    "Spacer": [np.nan] * max_len,
    
    "GTEx_Normal_log2_TPM": pad_list(gtex_log2, max_len),
    "TCGA_GBM_log2_TPM": pad_list(gbm_tumor_log2, max_len),
    "TCGA_LGG_log2_TPM": pad_list(lgg_tumor_log2, max_len)
}

df_prism = pd.DataFrame(prism_data)

# Save to CSV
df_prism.to_csv(output_file, index=False, na_rep="")
print(f"Prism-plottable CSV saved successfully to: {output_file}")

# Print summary statistics
print("\n--- Summary Statistics for SLC1A5 Expression (TPM) ---")
print(f"GTEx Normal    - N: {len(gtex_vals):3d}, Mean: {np.mean(gtex_vals):8.4f}, Median: {np.median(gtex_vals):8.4f}, Min: {np.min(gtex_vals):8.4f}, Max: {np.max(gtex_vals):8.4f}")
print(f"TCGA GBM Tumor - N: {len(gbm_tumor_vals):3d}, Mean: {np.mean(gbm_tumor_vals):8.4f}, Median: {np.median(gbm_tumor_vals):8.4f}, Min: {np.min(gbm_tumor_vals):8.4f}, Max: {np.max(gbm_tumor_vals):8.4f}")
print(f"TCGA LGG Tumor - N: {len(lgg_tumor_vals):3d}, Mean: {np.mean(lgg_tumor_vals):8.4f}, Median: {np.median(lgg_tumor_vals):8.4f}, Min: {np.min(lgg_tumor_vals):8.4f}, Max: {np.max(lgg_tumor_vals):8.4f}")

print("\n--- Summary Statistics for SLC1A5 Expression (log2(TPM+1)) ---")
print(f"GTEx Normal    - N: {len(gtex_log2):3d}, Mean: {np.mean(gtex_log2):8.4f}, Median: {np.median(gtex_log2):8.4f}, Min: {np.min(gtex_log2):8.4f}, Max: {np.max(gtex_log2):8.4f}")
print(f"TCGA GBM Tumor - N: {len(gbm_tumor_log2):3d}, Mean: {np.mean(gbm_tumor_log2):8.4f}, Median: {np.median(gbm_tumor_log2):8.4f}, Min: {np.min(gbm_tumor_log2):8.4f}, Max: {np.max(gbm_tumor_log2):8.4f}")
print(f"TCGA LGG Tumor - N: {len(lgg_tumor_log2):3d}, Mean: {np.mean(lgg_tumor_log2):8.4f}, Median: {np.median(lgg_tumor_log2):8.4f}, Min: {np.min(lgg_tumor_log2):8.4f}, Max: {np.max(lgg_tumor_log2):8.4f}")
