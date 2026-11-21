import os
import glob
import gzip
import requests
import json
import pandas as pd
import numpy as np

workspace_dir = r"d:\DATA\TIGER_analysis\TIGER_analysis"
gtex_file = os.path.join(workspace_dir, "GTEx_data", "gene_tpm_v10_brain_cerebellar_hemisphere.gct.gz")
excel_output = os.path.join(workspace_dir, "glioma_expression_master.xlsx")
csv_output = os.path.join(workspace_dir, "slc1a5_expression_prism.csv")

target_genes = ["PHGDH", "PSPH", "PSAT1", "SHMT1", "SHMT2", "MTHFD1", "MTHFD2", "SLC1A4", "SLC12A4", "SLC1A5"]
base_url = "https://www.cbioportal.org/api"

print("Step 1: Mapping gene symbols to cBioPortal Entrez Gene IDs...")
gene_to_entrez = {}
for gene in target_genes:
    url = f"{base_url}/genes/{gene}"
    r = requests.get(url)
    if r.status_code == 200:
        entrez_id = r.json().get("entrezGeneId")
        gene_to_entrez[gene] = entrez_id
    else:
        print(f"Error mapping {gene}")

print("\nStep 2: Fetching GBM and Diffuse Glioma from cBioPortal API...")
cbio_data = {
    "GBM": {gene: [] for gene in target_genes},
    "Diffuse": {gene: [] for gene in target_genes}
}

studies = {
    "GBM": {
        "profile_id": "gbm_tcga_gdc_mrna_seq_tpm",
        "sample_list_id": "gbm_tcga_gdc_all"
    },
    "Diffuse": {
        "profile_id": "difg_tcga_gdc_mrna_seq_tpm",
        "sample_list_id": "difg_tcga_gdc_all"
    }
}

for cohort, info in studies.items():
    for gene in target_genes:
        entrez_id = gene_to_entrez.get(gene)
        url = f"{base_url}/molecular-profiles/{info['profile_id']}/molecular-data?sampleListId={info['sample_list_id']}&entrezGeneId={entrez_id}"
        r = requests.get(url)
        if r.status_code == 200:
            vals = [dp.get("value") for dp in r.json() if dp.get("value") is not None]
            cbio_data[cohort][gene] = vals
            print(f"  - Cohort {cohort}, Gene {gene:7s}: fetched {len(vals)} values.")
        else:
            print(f"  - Error fetching {gene} for cohort {cohort}")

print("\nStep 3: Extracting LGG values from local GDC files...")
lgg_dir = os.path.join(workspace_dir, "GDCdata", "TCGA-LGG")
lgg_files = glob.glob(os.path.join(lgg_dir, "**", "*.tsv"), recursive=True)

# Map LGG sample types
uuid_to_meta = {}
all_uuids = [f.split(os.sep)[-2] for f in lgg_files]
batch_size = 100
for i in range(0, len(all_uuids), batch_size):
    batch = all_uuids[i:i+batch_size]
    url = "https://api.gdc.cancer.gov/files"
    params = {
        "filters": json.dumps({"op": "in", "content": {"field": "files.file_id", "value": batch}}),
        "fields": "file_id,cases.samples.sample_type",
        "format": "json",
        "size": str(len(batch))
    }
    r = requests.get(url, params=params)
    if r.status_code == 200:
        for hit in r.json().get("data", {}).get("hits", []):
            fid = hit.get("file_id")
            cases = hit.get("cases", [])
            if cases and cases[0].get("samples"):
                uuid_to_meta[fid] = cases[0]["samples"][0].get("sample_type")

local_lgg_tpm = {gene: [] for gene in target_genes}
target_set = set(target_genes)

for fpath in lgg_files:
    uuid = fpath.split(os.sep)[-2]
    if uuid_to_meta.get(uuid) != "Primary Tumor":
        continue
    
    file_vals = {}
    with open(fpath, "r") as f:
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
                gene_name = parts[name_idx]
                if gene_name in target_set:
                    file_vals[gene_name] = float(parts[tpm_idx])
                    if len(file_vals) == len(target_set):
                        break
    for gene in target_genes:
        if gene in file_vals:
            local_lgg_tpm[gene].append(file_vals[gene])

print(f"Extracted LGG values from {len(local_lgg_tpm[target_genes[0]])} files.")

print("\nStep 4: Extracting GTEx values...")
gtex_data = {gene: [] for gene in target_genes}
if os.path.exists(gtex_file):
    with gzip.open(gtex_file, "rt") as f:
        f.readline()
        f.readline()
        headers = f.readline().strip().split("\t")
        for row in f:
            parts = row.strip().split("\t")
            if len(parts) > 1:
                gene_symbol = parts[1]
                if gene_symbol in target_set:
                    vals = [float(val) for val in parts[2:]]
                    gtex_data[gene_symbol] = vals

print("\nStep 5: Compiling data and generating sheets...")
with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
    for gene in target_genes:
        gtex_vals = gtex_data.get(gene, [])
        gbm_vals = cbio_data["GBM"].get(gene, [])
        diffuse_vals = cbio_data["Diffuse"].get(gene, [])
        lgg_vals = local_lgg_tpm.get(gene, [])
        
        # Sort values
        gtex_raw = sorted(gtex_vals)
        gtex_log2 = [np.log2(val + 1) for val in gtex_raw]
        
        gbm_raw = sorted(gbm_vals)
        gbm_log2 = [np.log2(val + 1) for val in gbm_raw]
        
        diffuse_raw = sorted(diffuse_vals)
        diffuse_log2 = [np.log2(val + 1) for val in diffuse_raw]
        
        lgg_raw = sorted(lgg_vals)
        lgg_log2 = [np.log2(val + 1) for val in lgg_raw]
        
        # Max length
        max_len = max(len(gtex_raw), len(gbm_raw), len(diffuse_raw), len(lgg_raw))
        
        def pad_list(lst, target_len):
            return lst + [np.nan] * (target_len - len(lst))
            
        prism_data = {
            "GTEx_Normal_TPM": pad_list(gtex_raw, max_len),
            "TCGA_GBM_Tumor_TPM": pad_list(gbm_raw, max_len),
            "TCGA_DiffuseGlioma_Tumor_TPM": pad_list(diffuse_raw, max_len),
            "TCGA_LGG_Tumor_TPM": pad_list(lgg_raw, max_len),
            
            "Spacer": [np.nan] * max_len,
            
            "GTEx_Normal_Log2_(TPM+1)": pad_list(gtex_log2, max_len),
            "TCGA_GBM_Log2_(TPM+1)": pad_list(gbm_log2, max_len),
            "TCGA_DiffuseGlioma_Log2_(TPM+1)": pad_list(diffuse_log2, max_len),
            "TCGA_LGG_Log2_(TPM+1)": pad_list(lgg_log2, max_len),
        }
        
        df_gene = pd.DataFrame(prism_data)
        df_gene.to_excel(writer, sheet_name=gene, index=False, na_rep="")
        
        if gene == "SLC1A5":
            df_gene.to_csv(csv_output, index=False, na_rep="")

print(f"\nMaster Excel sheet saved successfully to: {excel_output}")

# Print summary statistics for all 10 genes (log2 values)
print("\n--- Summary Statistics (log2 values) for Genes in master sheet ---")
for gene in target_genes:
    g_log2 = [np.log2(val + 1) for val in gtex_data.get(gene, [])]
    gb_log2 = [np.log2(val + 1) for val in cbio_data["GBM"].get(gene, [])]
    lg_log2 = [np.log2(val + 1) for val in local_lgg_tpm.get(gene, [])]
    df_log2 = [np.log2(val + 1) for val in cbio_data["Diffuse"].get(gene, [])]
    
    print(f"\nGene: {gene}")
    print(f"  GTEx Normal    - N: {len(g_log2):3d}, Mean: {np.mean(g_log2):.4f}, Median: {np.median(g_log2):.4f}")
    print(f"  TCGA GBM Tumor - N: {len(gb_log2):3d}, Mean: {np.mean(gb_log2):.4f}, Median: {np.median(gb_log2):.4f}")
    print(f"  TCGA Diffuse   - N: {len(df_log2):3d}, Mean: {np.mean(df_log2):.4f}, Median: {np.median(df_log2):.4f}")
    print(f"  TCGA LGG Tumor - N: {len(lg_log2):3d}, Mean: {np.mean(lg_log2):.4f}, Median: {np.median(lg_log2):.4f}")
