# GBM Bioinformatic & Biostatistical Analysis

Analysis scripts for glioblastoma (GBM) transcriptomic, correlation, radiosensitivity index (RSI), and survival analyses across TCGA-GBM, GTEx normal brain tissue, and TIGER anti-PD-1 cohorts.

## Repository Structure

- **`correlation/`** — TCGA-GBM serine/one-carbon metabolism gene vs. immune marker correlation analyses (TPM & FPKM)
  - `analyze_correlation.py`
  - `generate_individual_plots.py`
  - `generate_prism_master_values.py`
- **`correlation_gtex/`** — GTEx normal brain tissue (cerebellar hemisphere) serine metabolism vs. immune marker correlation analysis
  - `analyze_gtex_correlation.py`
- **`RSI_radiation_survival/`** — Radiosensitivity Index (RSI) calculation, radiation-treated survival analysis, and 8-gene / 11-gene serine pathway correlation analyses in TCGA-GBM
  - `generate_rsi_data_and_stats.py`
  - `run_rsi_complete_analysis.py`
  - `generate_survminer_plots.R`
  - `plot_rsi_serine_correlation.py`
  - `plot_complete_11gene_correlation.py`
  - `generate_strict_11gene_analysis.py`
- **`TIGER_analysis/`** — Anti-PD-1 immunotherapy cohort (PRJNA482620 / TIGER) and TCGA-GBM serine score survival analyses
  - `extract_multi_genes_prism.py`
  - `extract_slc1a5_expression.py`
  - `serine_survival_analysis.R`
  - `patient_level_analysis/analyze_patient_level.py`
  - `pre_post_analysis/analyze_pre_post.py`
  - `tiger_webserver_replication/analyze_tiger_replication.py`
  - `tiger_webserver_replication/serine_replication_analysis.R`
