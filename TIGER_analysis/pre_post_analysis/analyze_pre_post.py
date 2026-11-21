# analyze_pre_post.py
# Date: 2026-06-09
# Purpose: Mapped analysis of pre vs. post anti-PD1 treatment gene expression
#          and clinical survival (KM curves) for target metabolic genes.

import os
import shutil
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Set premium plotting aesthetics
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

# 1. File Paths
clinical_file = 'GBM-PRJNA482620.Response_clinical.tsv.xlsx'
expression_file = 'GBM-PRJNA482620.Response.tsv'

# Check parent directory if not found locally
if not os.path.exists(clinical_file):
    clinical_file = os.path.join('..', clinical_file)
if not os.path.exists(expression_file):
    expression_file = os.path.join('..', expression_file)

if not os.path.exists(clinical_file) or not os.path.exists(expression_file):
    raise FileNotFoundError("Could not find clinical or expression data files.")

print(f"Loading clinical data from: {clinical_file}")
df_clin = pd.read_excel(clinical_file)

print(f"Loading expression data from: {expression_file}")
df_expr = pd.read_csv(expression_file, sep='\t')

# Target metabolic genes
target_genes = ['PHGDH', 'PSPH', 'PSAT1', 'SHMT1', 'SHMT2', 'SLC1A4', 'SLC1A5']

# 2. SRA Run Chronological Mapping
# Deduced from patient_id and surgery date from SRA database
run_mapping = {
    'SRR8281218': {'patient': 'Pt20', 'stage': 'Pre', 'date': '2016-02-16'},
    'SRR8281219': {'patient': 'Pt20', 'stage': 'Post', 'date': '2016-07-18'},
    'SRR8281220': {'patient': 'Pt21', 'stage': 'Post', 'date': '2016-01-20'},
    'SRR8281221': {'patient': 'Pt20', 'stage': 'Primary', 'date': '2014-08-11'},
    'SRR8281222': {'patient': 'Pt13', 'stage': 'Post', 'date': '2015-11-13'},
    'SRR8281223': {'patient': 'Pt21', 'stage': 'Pre', 'date': '2015-08-11'},
    'SRR8281224': {'patient': 'Pt13', 'stage': 'Primary', 'date': '2014-02-19'},
    'SRR8281225': {'patient': 'Pt13', 'stage': 'Pre', 'date': '2015-05-29'},
    'SRR8281226': {'patient': 'Pt53', 'stage': 'Pre', 'date': '2014-02-13'},
    'SRR8281227': {'patient': 'Pt53', 'stage': 'Post_rep', 'date': '2016-01-20'},
    'SRR8281228': {'patient': 'Pt3',  'stage': 'Post', 'date': '2015-07-07'},
    'SRR8281229': {'patient': 'Pt24', 'stage': 'Pre', 'date': '2012-05-09'},
    'SRR8281230': {'patient': 'Pt56', 'stage': 'Post', 'date': '2015-11-04'},
    'SRR8281231': {'patient': 'Pt3',  'stage': 'Pre', 'date': '2015-01-29'},
    'SRR8281232': {'patient': 'Pt69', 'stage': 'Post', 'date': '2016-01-06'},
    'SRR8281233': {'patient': 'Pt56', 'stage': 'Pre', 'date': '2015-01-14'},
    'SRR8281234': {'patient': 'Pt46', 'stage': 'Post', 'date': '2014-01-21'},
    'SRR8281235': {'patient': 'Pt69', 'stage': 'Pre', 'date': '2015-05-18'},
    'SRR8281236': {'patient': 'Pt53', 'stage': 'Post', 'date': '2016-01-20'},
    'SRR8281237': {'patient': 'Pt46', 'stage': 'Pre', 'date': '2013-06-24'},
    'SRR8281238': {'patient': 'Pt102', 'stage': 'Post', 'date': '2016-10-31'},
    'SRR8281239': {'patient': 'Pt103', 'stage': 'Post', 'date': '2016-11-22'},
    'SRR8281240': {'patient': 'Pt104', 'stage': 'Pre', 'date': '2014-01-15'},
    'SRR8281241': {'patient': 'Pt104', 'stage': 'Post', 'date': '2016-12-08'},
    'SRR8281242': {'patient': 'Pt101', 'stage': 'Post_rep', 'date': '2016-06-15'},
    'SRR8281243': {'patient': 'Pt101', 'stage': 'Pre', 'date': '2012-11-27'},
    'SRR8281244': {'patient': 'Pt100', 'stage': 'Post', 'date': '2016-07-05'},
    'SRR8281245': {'patient': 'Pt100', 'stage': 'Pre', 'date': '2015-08-19'},
    'SRR8281246': {'patient': 'Pt71', 'stage': 'Post', 'date': '2016-01-27'},
    'SRR8281247': {'patient': 'Pt55', 'stage': 'Post', 'date': '2014-10-08'},
    'SRR8281248': {'patient': 'Pt55', 'stage': 'Pre', 'date': '2014-06-18'},
    'SRR8281249': {'patient': 'Pt25', 'stage': 'Pre', 'date': '2015-03-24'},
    'SRR8281250': {'patient': 'Pt102', 'stage': 'Pre', 'date': '2015-04-13'},
    'SRR8281251': {'patient': 'Pt101', 'stage': 'Post', 'date': '2016-06-15'}
}

# 3. Extract target gene expression
print("Extracting target gene expression levels...")
df_expr_target = df_expr[df_expr['GENE_SYMBOL'].isin(target_genes)].set_index('GENE_SYMBOL')
expr_df = df_expr_target.T

# Map SRA runs to metadata
mapping_df = pd.DataFrame.from_dict(run_mapping, orient='index')
merged_df = expr_df.merge(mapping_df, left_index=True, right_index=True)

# 4. Clean Replicates
# Standardize 'Post_rep' as 'Post' and average duplicate runs for patient/stage
merged_df['stage_clean'] = merged_df['stage'].replace('Post_rep', 'Post')

# Save raw mapped run data
merged_df.index.name = 'sample_id'
merged_df.to_csv('pre_post_expression_data_raw.csv')

# Aggregate expression (mean) per patient and stage
expression_agg = merged_df.groupby(['patient', 'stage_clean'])[target_genes].mean().reset_index()

# 5. Paired Pre vs. Post Analysis (N = 13 patients)
pivoted = expression_agg.pivot(index='patient', columns='stage_clean', values=target_genes)
# Filter for patients with both Pre and Post
paired_patients = pivoted.dropna(subset=[(gene, 'Pre') for gene in target_genes] + [(gene, 'Post') for gene in target_genes]).index
pivoted_paired = pivoted.loc[paired_patients]
print(f"Number of paired patients found: {len(pivoted_paired)}")

# Save processed paired matrix
pivoted_paired_flat = pivoted_paired.copy()
pivoted_paired_flat.columns = [f"{gene}_{stage}" for gene, stage in pivoted_paired_flat.columns]
pivoted_paired_flat.to_csv('pre_post_expression_data.csv')

# Perform paired stats
stats_records = []
for gene in target_genes:
    pre_vals = pivoted_paired[(gene, 'Pre')]
    post_vals = pivoted_paired[(gene, 'Post')]
    
    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(post_vals, pre_vals)
    # Wilcoxon signed-rank test
    try:
        w_stat, w_pval = stats.wilcoxon(post_vals, pre_vals)
    except Exception as e:
        w_stat, w_pval = np.nan, np.nan
        
    mean_pre = pre_vals.mean()
    mean_post = post_vals.mean()
    mean_diff = mean_post - mean_pre
    
    stats_records.append({
        'Gene': gene,
        'Mean_Pre': mean_pre,
        'Mean_Post': mean_post,
        'Mean_Diff': mean_diff,
        't_stat': t_stat,
        't_pval': t_pval,
        'wilcox_stat': w_stat,
        'wilcox_pval': w_pval
    })

df_stats = pd.DataFrame(stats_records)
df_stats.to_csv('paired_stats_summary.csv', index=False)

print("\n--- Paired Pre vs. Post Statistical Results ---")
print(df_stats.to_string(index=False))

# 6. Plot Paired Expression Boxplots with Lines
print("\nGenerating paired expression plots...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()

# Tones: soft teal for Pre, soft coral for Post
pre_color = '#007A87'
post_color = '#E05D5D'
line_color = '#7f8c8d'

for idx, gene in enumerate(target_genes):
    ax = axes_flat[idx]
    pre_vals = pivoted_paired[(gene, 'Pre')].values
    post_vals = pivoted_paired[(gene, 'Post')].values
    
    # Boxplots
    sns.boxplot(data=[pre_vals, post_vals], ax=ax, palette=[pre_color, post_color], width=0.4, showfliers=False, boxprops=dict(alpha=0.3))
    
    # Individual connecting lines
    for i in range(len(pre_vals)):
        ax.plot([0, 1], [pre_vals[i], post_vals[i]], color=line_color, alpha=0.6, marker='o', markersize=6, lw=1.2)
        
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'], fontsize=11, fontweight='medium')
    ax.set_ylabel('Expression Level (log2 FPKM+1)', fontsize=11)
    ax.set_title(f'{gene}', fontsize=14, fontweight='bold', pad=10)
    
    # Annotate p-values
    w_pval = df_stats.loc[df_stats['Gene'] == gene, 'wilcox_pval'].values[0]
    t_pval = df_stats.loc[df_stats['Gene'] == gene, 't_pval'].values[0]
    p_text = f"Paired t-test: p = {t_pval:.4f}\nWilcoxon: p = {w_pval:.4f}"
    
    # Apply bold title style if Wilcoxon or t-test p < 0.05
    if w_pval < 0.05 or t_pval < 0.05:
        ax.title.set_color('#BC3C29')
        ax.title.set_text(f"{gene} *")
        
    ax.text(0.05, 0.05, p_text, transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E2E8F0', alpha=0.9))
    sns.despine(ax=ax, offset=5)

# 8th subplot: Summary Table of Stats
ax_table = axes_flat[7]
ax_table.axis('off')
table_data = df_stats[['Gene', 'Mean_Diff', 't_pval', 'wilcox_pval']].copy()
table_data.columns = ['Gene', 'Mean Diff', 't-test p', 'Wilcox p']
table_data['Mean Diff'] = table_data['Mean Diff'].round(3)
table_data['t-test p'] = table_data['t-test p'].apply(lambda x: f'{x:.4f}')
table_data['Wilcox p'] = table_data['Wilcox p'].apply(lambda x: f'{x:.4f}')

the_table = ax_table.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(11)
the_table.scale(1.1, 1.8)

# Format header cell colors
for j, label in enumerate(table_data.columns):
    cell = the_table[0, j]
    cell.set_text_props(weight='bold', color='white')
    cell.set_facecolor('#333333')
    
ax_table.set_title('Pre vs. Post Summary', fontsize=14, fontweight='bold', pad=10)

fig.suptitle('Glioblastoma Anti-PD1 Paired Gene Expression (N=13 patients)\nZhao et al. 2019 (Nature Medicine)', 
             fontsize=18, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93])

fig.savefig('paired_expression_plots.png', dpi=300)
fig.savefig('paired_expression_plots.pdf')
plt.close(fig)

# 7. Kaplan-Meier Survival Analysis on Pre-treatment Baseline (N = 15 patients)
print("\nPerforming survival analysis on pre-treatment baseline samples...")
# Filter clinical survival data
df_clin_clean = df_clin[['dataset_id', 'overall survival (days)', 'vital status']].drop_duplicates().set_index('dataset_id')
df_clin_clean['Status'] = (df_clin_clean['vital status'] == 'Dead').astype(int)
df_clin_clean['Time'] = df_clin_clean['overall survival (days)']

# Extract pre-treatment expression levels (15 patients have Pre samples)
pre_expr_df = merged_df[merged_df['stage'] == 'Pre'][['patient'] + target_genes].set_index('patient')
survival_df = pre_expr_df.merge(df_clin_clean, left_index=True, right_index=True)
survival_df['OS_months'] = survival_df['Time'] / 30.4375

print(f"Number of patients with baseline survival data: {len(survival_df)}")

survival_stats = []
fig_surv, axes_surv = plt.subplots(2, 4, figsize=(20, 10))
axes_surv_flat = axes_surv.flatten()

high_surv_color = '#BC3C29'  # NEJM Red
low_surv_color = '#0072B5'   # NEJM Blue

for idx, gene in enumerate(target_genes):
    ax = axes_surv_flat[idx]
    
    # Stratify by median split of pre-treatment expression
    median_val = survival_df[gene].median()
    survival_df[f'{gene}_Group'] = np.where(survival_df[gene] > median_val, 'High', 'Low')
    
    high_group = survival_df[survival_df[f'{gene}_Group'] == 'High']
    low_group = survival_df[survival_df[f'{gene}_Group'] == 'Low']
    
    # Log-rank test
    res = logrank_test(
        high_group['OS_months'], low_group['OS_months'],
        event_observed_A=high_group['Status'], event_observed_B=low_group['Status']
    )
    p_val = res.p_value
    
    # Fit survival curves
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    kmf_high.fit(high_group['OS_months'], high_group['Status'], label=f"High (N={len(high_group)})")
    kmf_low.fit(low_group['OS_months'], low_group['Status'], label=f"Low (N={len(low_group)})")
    
    kmf_high.plot_survival_function(ax=ax, color=high_surv_color, lw=2.5, ci_show=False, 
                                    show_censors=True, censor_styles={'ms': 8, 'mew': 1.5, 'marker': '+'})
    kmf_low.plot_survival_function(ax=ax, color=low_surv_color, lw=2.5, ci_show=False, 
                                   show_censors=True, censor_styles={'ms': 8, 'mew': 1.5, 'marker': '+'})
    
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 60)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_xlabel('Overall Survival (Months)', fontsize=11)
    ax.set_title(f'{gene} (Pre-treatment)', fontsize=14, fontweight='bold', pad=10)
    
    p_str = f"p = {p_val:.4f}" if p_val >= 0.001 else f"p = {p_val:.2e}"
    if p_val < 0.05:
        ax.title.set_color('#BC3C29')
        ax.title.set_text(f"{gene} (Pre-treatment) *")
        
    ax.text(0.05, 0.1, f"Log-rank {p_str}", transform=ax.transAxes, fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#E2E8F0', alpha=0.9))
    sns.despine(ax=ax, offset=5)
    
    # Median survival calculation
    med_high = kmf_high.median_survival_time_
    med_low = kmf_low.median_survival_time_
    
    survival_stats.append({
        'Gene': gene,
        'Median_Expression': median_val,
        'LogRank_Stat': res.test_statistic,
        'LogRank_pVal': p_val,
        'Median_OS_High_Months': med_high,
        'Median_OS_Low_Months': med_low
    })

df_surv_stats = pd.DataFrame(survival_stats)
df_surv_stats.to_csv('survival_stats_summary.csv', index=False)

print("\n--- Kaplan-Meier Survival Analysis Results ---")
print(df_surv_stats.to_string(index=False))

# 8th subplot: Summary Table of Survival Stats
ax_table_surv = axes_surv_flat[7]
ax_table_surv.axis('off')
table_surv_data = df_surv_stats[['Gene', 'LogRank_pVal', 'Median_OS_High_Months', 'Median_OS_Low_Months']].copy()
table_surv_data.columns = ['Gene', 'Log-rank p', 'Med OS High', 'Med OS Low']
table_surv_data['Log-rank p'] = table_surv_data['Log-rank p'].apply(lambda x: f'{x:.4f}')

def format_med(val):
    if np.isnan(val) or np.isinf(val):
        return 'N/A'
    return f'{val:.1f} mo'

table_surv_data['Med OS High'] = table_surv_data['Med OS High'].apply(format_med)
table_surv_data['Med OS Low'] = table_surv_data['Med OS Low'].apply(format_med)

the_surv_table = ax_table_surv.table(cellText=table_surv_data.values, colLabels=table_surv_data.columns, loc='center', cellLoc='center')
the_surv_table.auto_set_font_size(False)
the_surv_table.set_fontsize(11)
the_surv_table.scale(1.1, 1.8)

# Format header cell colors
for j, label in enumerate(table_surv_data.columns):
    cell = the_surv_table[0, j]
    cell.set_text_props(weight='bold', color='white')
    cell.set_facecolor('#333333')

ax_table_surv.set_title('Survival Summary', fontsize=14, fontweight='bold', pad=10)

fig_surv.suptitle('Kaplan-Meier Curves Stratified by Pre-treatment Gene Expression (N=15 patients)\nZhao et al. 2019 (Nature Medicine)', 
                  fontsize=18, fontweight='bold', y=0.98)
fig_surv.tight_layout(rect=[0, 0, 1, 0.93])

fig_surv.savefig('survival_km_curves.png', dpi=300)
fig_surv.savefig('survival_km_curves.pdf')
plt.close(fig_surv)

print("\nAnalysis complete! All files generated.")
