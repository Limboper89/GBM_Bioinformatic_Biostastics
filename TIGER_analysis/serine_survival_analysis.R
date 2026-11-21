# serine_survival_analysis.R
# Date: 2026-06-09
# Purpose: Calculate Serine Score and perform survival analysis for TCGA-GBM.

cat("Loading required libraries...\n")
library(TCGAbiolinks)
library(SummarizedExperiment)
library(GSVA)
library(survminer)
library(survival)
library(ggplot2)

# Set output paths
project_dir <- "d:/DATA/TIGER_analysis/TIGER_analysis"
pdf_path <- file.path(project_dir, "serine_score_survival_km_plot.pdf")
png_path <- file.path(project_dir, "serine_score_survival_km_plot.png")

cat("Step 1: Querying GDC database for TCGA-GBM Transcriptome Profiling...\n")
query <- GDCquery(
  project = "TCGA-GBM",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",
  sample.type = "Primary Tumor"
)

cat("Step 2: Downloading data...\n")
GDCdownload(query)

cat("Step 3: Preparing and loading SummarizedExperiment object...\n")
data <- GDCprepare(query)

cat("Step 4: Processing clinical data...\n")
clinical_data <- as.data.frame(colData(data))

# Print initial sample count
cat(sprintf("Loaded expression data for %d samples.\n", ncol(data)))

# Clean clinical survival data
# Vital status mapping: Dead -> 1, Alive -> 0
clinical_data$status <- ifelse(clinical_data$vital_status == "Dead", 1, 0)

# Survival time: days to death if dead, days to last follow up if alive
clinical_data$time_days <- ifelse(
  clinical_data$vital_status == "Dead",
  clinical_data$days_to_death,
  clinical_data$days_to_last_follow_up
)

# Convert time to months
clinical_data$time <- clinical_data$time_days / 30.4375

# Filter out samples with missing or non-positive survival time, or missing status
valid_idx <- !is.na(clinical_data$time) & clinical_data$time > 0 & !is.na(clinical_data$status)
clinical_clean <- clinical_data[valid_idx, ]
cat(sprintf("Filtered to %d samples with valid survival data.\n", nrow(clinical_clean)))

# Align expression data with clean clinical data
expr_matrix <- assay(data, "fpkm_unstrand")
expr_clean <- expr_matrix[, rownames(clinical_clean), drop = FALSE]

cat("Step 5: Processing expression data and mapping gene symbols...\n")
row_info <- as.data.frame(rowData(data))
row_info$gene_name <- as.character(row_info$gene_name)

# Filter out genes without names
valid_gene_idx <- !is.na(row_info$gene_name) & row_info$gene_name != ""
all_genes_matrix <- expr_clean[valid_gene_idx, , drop = FALSE]
all_gene_names <- row_info$gene_name[valid_gene_idx]

# Average duplicates using high-performance rowsum method
cat("Handling duplicate gene names...\n")
all_genes_matrix_clean <- rowsum(all_genes_matrix, all_gene_names)
gene_counts <- as.vector(table(all_gene_names))
all_genes_matrix_clean <- all_genes_matrix_clean / gene_counts

# Log2-transform the cleaned expression matrix (FPKM)
log_expr_matrix <- log2(all_genes_matrix_clean + 1)

# Define the serine pathway gene signature
serine_signature <- c("PHGDH", "PSAT1", "PSPH", "SHMT1", "SHMT2", "SLC1A4", "SLC1A5", "SLC12A4")

# Ensure all signature genes are in our expression matrix
serine_genes_in_data <- intersect(serine_signature, rownames(log_expr_matrix))
cat("Serine genes found in the dataset:\n")
print(serine_genes_in_data)

if (length(serine_genes_in_data) == 0) {
  stop("None of the serine pathway genes were found in the dataset! Check rowData mapping.")
}

gene_sets <- list(Serine_Pathway = serine_genes_in_data)

cat("Step 6: Calculating Serine Score using ssGSEA...\n")
param <- ssgseaParam(exprData = as.matrix(log_expr_matrix), geneSets = gene_sets)
ssgsea_results <- gsva(param)

# Extract scores
serine_scores <- ssgsea_results["Serine_Pathway", ]

# Combine clinical data and serine score
survival_df <- data.frame(
  time = clinical_clean$time,
  status = clinical_clean$status,
  Serine_Score = serine_scores[rownames(clinical_clean)]
)

cat("Step 7: Identifying optimal cutpoint using surv_cutpoint...\n")
cutpoint_obj <- surv_cutpoint(
  data = survival_df,
  time = "time",
  event = "status",
  variables = "Serine_Score"
)

cutpoint_val <- cutpoint_obj$cutpoint$cutpoint
cat(sprintf("Optimal Serine Score cutpoint: %f\n", cutpoint_val))

# Stratify patients into High and Low Serine groups
survival_df$Group <- factor(
  ifelse(survival_df$Serine_Score > cutpoint_val, "High Serine", "Low Serine"),
  levels = c("Low Serine", "High Serine") # Low Serine will be the reference group
)

cat("Patient count in groups:\n")
print(table(survival_df$Group))

cat("Step 8: Performing survival curve fitting and plotting...\n")
fit <- survfit(Surv(time, status) ~ Group, data = survival_df)

# Run log-rank test to get details
sdiff <- survdiff(Surv(time, status) ~ Group, data = survival_df)
p_value <- 1 - pchisq(sdiff$chisq, length(sdiff$n) - 1)
cat(sprintf("Log-rank p-value: %e\n", p_value))

# Calculate median survival times
fit_summary <- summary(fit)$table
cat("Survival statistics:\n")
print(fit_summary)

# Plot Kaplan-Meier curve using ggsurvplot
p_val_formatted <- sprintf("p = %.3f (Log-rank test)", p_value)
p <- ggsurvplot(
  fit,
  data = survival_df,
  pval = p_val_formatted,   # Show exact formatted p-value
  pval.size = 4.5,          # Publication font size
  pval.coord = c(5, 0.15),  # Positioned cleanly in the lower-left empty space
  risk.table = TRUE,
  risk.table.col = "strata",
  risk.table.y.text = FALSE, # Hide redundant y-axis text in risk table for a clean look
  legend.labs = c("Low Serine", "High Serine"),
  legend.title = "Serine Score Group",
  palette = c("#0072B5", "#BC3C29"), # NEJM colors: Blue for Low Serine, Red for High Serine
  title = "TCGA-GBM Kaplan-Meier Survival Analysis",
  subtitle = "Stratified by ssGSEA Serine Score (Optimal Cutpoint)",
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  censor.size = 4.5,        # Adjust censor tick size
  size = 1,                 # Line thickness
  ggtheme = theme_classic(base_size = 12) + # Classic publication theme (no gridlines)
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 11),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 11, face = "bold"),
      legend.position = "top"
    )
)

cat(sprintf("Step 9: Saving plot to PDF: %s\n", pdf_path))
pdf(pdf_path, width = 8, height = 7, onefile = FALSE)
print(p)
dev.off()

cat(sprintf("Saving plot to PNG: %s\n", png_path))
png(png_path, width = 800, height = 700, res = 100)
print(p)
dev.off()

cat("Analysis complete! All files successfully generated.\n")
