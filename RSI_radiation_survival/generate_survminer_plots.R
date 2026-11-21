suppressPackageStartupMessages({
  library(survival)
  library(survminer)
  library(ggplot2)
})

work_dir <- "d:/DATA/GBM_bioinormatics/GBM_analysis"
tables_dir <- file.path(work_dir, "RSI_radiation_survival", "tables")
plots_dir <- file.path(work_dir, "RSI_radiation_survival", "plots")

# Load master dataset
df_path <- file.path(tables_dir, "TCGA_GBM_RSI_patient_master.csv")
df <- read.csv(df_path, stringsAsFactors = FALSE)

# Convert boolean columns properly
df$has_rad <- as.logical(df$has_rad)
df$has_immuno <- as.logical(df$has_immuno)

# Colors: NEJM/Lancet style
palette_nejm <- c("#0072B5", "#BC3C29") # Blue for Low/Sensitive, Red for High/Resistant

# -------------------------------------------------------------
# 1. Radiation Therapy Only (Median Cutpoint)
# -------------------------------------------------------------
df_rad <- df[df$treatment_group == "Radiation Only", ]
med_rad <- median(df_rad$RSI)
df_rad$Group_Median <- factor(
  ifelse(df_rad$RSI >= med_rad, "High RSI (Radioresistant)", "Low RSI (Radiosensitive)"),
  levels = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)")
)

fit_rad_med <- survfit(Surv(time_months, status) ~ Group_Median, data = df_rad)
sdiff_rad_med <- survdiff(Surv(time_months, status) ~ Group_Median, data = df_rad)
p_rad_med <- 1 - pchisq(sdiff_rad_med$chisq, length(sdiff_rad_med$n) - 1)

p1 <- ggsurvplot(
  fit_rad_med,
  data = df_rad,
  pval = sprintf("Log-rank p = %.3f", p_rad_med),
  pval.size = 4.5,
  pval.coord = c(1, 0.15),
  risk.table = TRUE,
  risk.table.col = "strata",
  risk.table.y.text = FALSE,
  legend.labs = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)"),
  legend.title = "RSI Stratum",
  palette = palette_nejm,
  title = "TCGA-GBM Radiation Therapy Only Cohort (N = 17)",
  subtitle = sprintf("Stratified by Median RSI (Cutpoint = %.4f)", med_rad),
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  censor.size = 4.5,
  size = 1.1,
  xlim = c(0, 32),
  break.time.by = 6,
  ggtheme = theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
      plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 10.5),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10.5, face = "bold"),
      legend.position = "top"
    )
)

png(file.path(plots_dir, "06b_KM_survminer_radiation_only_median.png"), width = 2400, height = 2100, res = 300)
print(p1)
dev.off()

pdf(file.path(plots_dir, "06b_KM_survminer_radiation_only_median.pdf"), width = 8, height = 7)
print(p1)
dev.off()

# -------------------------------------------------------------
# 2. Radiation Therapy Only (Optimal Cutpoint = 0.8630)
# -------------------------------------------------------------
opt_rad <- 0.8630
df_rad$Group_Opt <- factor(
  ifelse(df_rad$RSI > opt_rad, "High RSI (Radioresistant)", "Low RSI (Radiosensitive)"),
  levels = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)")
)

fit_rad_opt <- survfit(Surv(time_months, status) ~ Group_Opt, data = df_rad)
sdiff_rad_opt <- survdiff(Surv(time_months, status) ~ Group_Opt, data = df_rad)
p_rad_opt <- 1 - pchisq(sdiff_rad_opt$chisq, length(sdiff_rad_opt$n) - 1)

p2 <- ggsurvplot(
  fit_rad_opt,
  data = df_rad,
  pval = sprintf("Log-rank p = %.3f", p_rad_opt),
  pval.size = 4.5,
  pval.coord = c(1, 0.15),
  risk.table = TRUE,
  risk.table.col = "strata",
  risk.table.y.text = FALSE,
  legend.labs = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)"),
  legend.title = "RSI Stratum",
  palette = palette_nejm,
  title = "TCGA-GBM Radiation Therapy Only Cohort (N = 17)",
  subtitle = sprintf("Stratified by Optimal Cutpoint (Cutpoint = %.4f)", opt_rad),
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  censor.size = 4.5,
  size = 1.1,
  xlim = c(0, 32),
  break.time.by = 6,
  ggtheme = theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
      plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 10.5),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10.5, face = "bold"),
      legend.position = "top"
    )
)

png(file.path(plots_dir, "07b_KM_survminer_radiation_only_optimal.png"), width = 2400, height = 2100, res = 300)
print(p2)
dev.off()

pdf(file.path(plots_dir, "07b_KM_survminer_radiation_only_optimal.pdf"), width = 8, height = 7)
print(p2)
dev.off()

# -------------------------------------------------------------
# 3. All Radiation Treated (Non-Immunotherapy, N = 201)
# -------------------------------------------------------------
df_all_rad <- df[!is.na(df$has_rad) & df$has_rad == TRUE & !is.na(df$has_immuno) & df$has_immuno == FALSE, ]
med_all <- median(df_all_rad$RSI)
df_all_rad$Group_Median <- factor(
  ifelse(df_all_rad$RSI >= med_all, "High RSI (Radioresistant)", "Low RSI (Radiosensitive)"),
  levels = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)")
)

fit_all_med <- survfit(Surv(time_months, status) ~ Group_Median, data = df_all_rad)
sdiff_all_med <- survdiff(Surv(time_months, status) ~ Group_Median, data = df_all_rad)
p_all_med <- 1 - pchisq(sdiff_all_med$chisq, length(sdiff_all_med$n) - 1)

p3 <- ggsurvplot(
  fit_all_med,
  data = df_all_rad,
  pval = sprintf("Log-rank p = %.3f", p_all_med),
  pval.size = 4.5,
  pval.coord = c(2, 0.15),
  risk.table = TRUE,
  risk.table.col = "strata",
  risk.table.y.text = FALSE,
  legend.labs = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)"),
  legend.title = "RSI Stratum",
  palette = palette_nejm,
  title = "All Radiation-Treated TCGA-GBM Cohort (Non-Immuno, N = 201)",
  subtitle = sprintf("Stratified by Median RSI (Cutpoint = %.4f)", med_all),
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  censor.size = 4.5,
  size = 1.1,
  xlim = c(0, 65),
  break.time.by = 12,
  ggtheme = theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
      plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 10.5),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10.5, face = "bold"),
      legend.position = "top"
    )
)

png(file.path(plots_dir, "08b_KM_survminer_all_radiation_median.png"), width = 2400, height = 2100, res = 300)
print(p3)
dev.off()

pdf(file.path(plots_dir, "08b_KM_survminer_all_radiation_median.pdf"), width = 8, height = 7)
print(p3)
dev.off()

# -------------------------------------------------------------
# 4. All Radiation Treated (Optimal Cutpoint = 0.5680)
# -------------------------------------------------------------
opt_all <- 0.5680
df_all_rad$Group_Opt <- factor(
  ifelse(df_all_rad$RSI > opt_all, "High RSI (Radioresistant)", "Low RSI (Radiosensitive)"),
  levels = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)")
)

fit_all_opt <- survfit(Surv(time_months, status) ~ Group_Opt, data = df_all_rad)
sdiff_all_opt <- survdiff(Surv(time_months, status) ~ Group_Opt, data = df_all_rad)
p_all_opt <- 1 - pchisq(sdiff_all_opt$chisq, length(sdiff_all_opt$n) - 1)

p4 <- ggsurvplot(
  fit_all_opt,
  data = df_all_rad,
  pval = sprintf("Log-rank p = %.3f", p_all_opt),
  pval.size = 4.5,
  pval.coord = c(2, 0.15),
  risk.table = TRUE,
  risk.table.col = "strata",
  risk.table.y.text = FALSE,
  legend.labs = c("Low RSI (Radiosensitive)", "High RSI (Radioresistant)"),
  legend.title = "RSI Stratum",
  palette = palette_nejm,
  title = "All Radiation-Treated TCGA-GBM Cohort (Non-Immuno, N = 201)",
  subtitle = sprintf("Stratified by Optimal Cutpoint (Cutpoint = %.4f)", opt_all),
  xlab = "Overall Survival (Months)",
  ylab = "Survival Probability",
  censor.size = 4.5,
  size = 1.1,
  xlim = c(0, 65),
  break.time.by = 12,
  ggtheme = theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
      plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 10.5),
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 10.5, face = "bold"),
      legend.position = "top"
    )
)

png(file.path(plots_dir, "09b_KM_survminer_all_radiation_optimal.png"), width = 2400, height = 2100, res = 300)
print(p4)
dev.off()

pdf(file.path(plots_dir, "09b_KM_survminer_all_radiation_optimal.pdf"), width = 8, height = 7)
print(p4)
dev.off()

cat("All survminer plots generated successfully!\n")
