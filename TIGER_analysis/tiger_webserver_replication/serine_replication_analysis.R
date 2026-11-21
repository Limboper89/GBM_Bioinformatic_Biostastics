# serine_replication_analysis.R
# Date: 2026-06-09
# Purpose: Calculate Serine Score and perform survival analysis for TIGER replication data (GBM anti-PD1 cohort).

cat("Loading required libraries...\n")
library(GSVA)
library(survminer)
library(survival)
library(ggplot2)

# Set input paths
csv_path <- "d:/DATA/TIGER_analysis/TIGER_analysis/tiger_webserver_replication/gbm_anti_pd1_tiger_master.csv"
tsv_path <- "d:/DATA/TIGER_analysis/TIGER_analysis/GBM-PRJNA482620.Response.tsv"

# Set output paths
project_dir <- "d:/DATA/TIGER_analysis/TIGER_analysis/tiger_webserver_replication"

pdf_proj <- file.path(project_dir, "replication_serine_score_survival_km_plot.pdf")
png_proj <- file.path(project_dir, "replication_serine_score_survival_km_plot.png")

cat("Step 1: Loading TIGER webserver replication datasets...\n")
clinical_data <- read.csv(csv_path)

# Filter out patients with missing or non-positive survival time
clinical_data <- clinical_data[!is.na(clinical_data$Time) & clinical_data$Time > 0 & !is.na(clinical_data$Status), ]
cat(sprintf("Loaded clinical data for %d patients.\n", nrow(clinical_data)))

# Load full transcriptome TSV
expression_data <- read.delim(tsv_path, header = TRUE)

# Extract gene symbols and create expression matrix
gene_names <- expression_data$GENE_SYMBOL
expr_matrix <- as.matrix(expression_data[, -1])
rownames(expr_matrix) <- gene_names

# Align columns (samples) with clinical patient names
common_samples <- intersect(clinical_data$patient_name, colnames(expr_matrix))
clinical_clean <- clinical_data[clinical_data$patient_name %in% common_samples, ]
expr_clean <- expr_matrix[, clinical_clean$patient_name, drop = FALSE]

cat(sprintf("Aligned data for %d common patients.\n", nrow(clinical_clean)))

# Average duplicate gene names if any (using rowsum)
if (any(duplicated(rownames(expr_clean)))) {
  cat("Handling duplicate gene names...\n")
  unique_genes <- rownames(expr_clean)
  expr_clean <- rowsum(expr_clean, unique_genes)
  gene_counts <- as.vector(table(unique_genes))
  expr_clean <- expr_clean / gene_counts
}

# Log2-transform expression data
log_expr_matrix <- log2(expr_clean + 1)

# Define serine pathway signature (incorporating SLC1A5 and SLC12A4)
serine_signature <- c("PHGDH", "PSAT1", "PSPH", "SHMT1", "SHMT2", "SLC1A4", "SLC1A5", "SLC12A4")

# Ensure all genes are present in the matrix
serine_genes_in_data <- intersect(serine_signature, rownames(log_expr_matrix))
cat("Serine signature genes present in replication dataset:\n")
print(serine_genes_in_data)

gene_sets <- list(Serine_Pathway = serine_genes_in_data)

cat("Step 2: Calculating Serine Score using ssGSEA...\n")
param <- ssgseaParam(exprData = as.matrix(log_expr_matrix), geneSets = gene_sets)
ssgsea_results <- gsva(param)
serine_scores <- ssgsea_results["Serine_Pathway", ]

# Combine clinical data and serine score
# Convert overall survival from days to months
survival_df <- data.frame(
  time = clinical_clean$Time / 30.4375,
  status = clinical_clean$Status,
  Serine_Score = serine_scores[clinical_clean$patient_name]
)
rownames(survival_df) <- clinical_clean$patient_name

cat("Step 3: Identifying optimal cutpoint using surv_cutpoint...\n")
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
  levels = c("Low Serine", "High Serine")
)

cat("Patient count in groups:\n")
print(table(survival_df$Group))

cat("Step 4: Performing survival curve fitting and plotting...\n")
fit <- survfit(Surv(time, status) ~ Group, data = survival_df)

# Run log-rank test
sdiff <- survdiff(Surv(time, status) ~ Group, data = survival_df)
p_value <- 1 - pchisq(sdiff$chisq, length(sdiff$n) - 1)
cat(sprintf("Log-rank p-value: %e\n", p_value))

fit_summary <- summary(fit)$table
cat("Survival statistics:\n")
print(fit_summary)

# Plot Kaplan-Meier curve using ggsurvplot (NEJM Style)
p_val_formatted <- sprintf("p = %.3f (Log-rank test)", p_value)
p <- ggsurvplot(
  fit,
  data = survival_df,
  pval = p_val_formatted,
  pval.size = 4.5,
  pval.coord = c(5, 0.15),
  risk.table = TRUE,
  risk.table.col = "strata",
  risk.table.y.text = FALSE,
  legend.labs = c("Low Serine", "High Serine"),
  legend.title = "Serine Score Group",
  palette = c("#0072B5", "#BC3C29"), # NEJM colors
  title = "TIGER Replication GBM (Anti-PD1) Survival Analysis",
  subtitle = "Stratified by ssGSEA Serine Score (Optimal Cutpoint)",
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  censor.size = 4.5,
  size = 1,
  ggtheme = theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 11),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 11, face = "bold"),
      legend.position = "top"
    )
)

cat("Step 5: Exporting plot to PDF & PNG formats...\n")
pdf(pdf_proj, width = 8, height = 7, onefile = FALSE)
print(p)
dev.off()

png(png_proj, width = 800, height = 700, res = 100)
print(p)
dev.off()

cat("Analysis complete! All files successfully generated.\n")
