"""
Integrate GDSC Drug Response Data with Methylation Features
============================================================

This script combines GSE68379 methylation data with GDSC drug response data
to create a comprehensive pharmacoepigenomics ML-ready dataset.

Author: Claude (Computational Biologist)
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set random seed for reproducibility
np.random.seed(42)

# Display settings
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

print("=" * 80)
print("GDSC DRUG RESPONSE INTEGRATION")
print("=" * 80)

# ============================================================================
# 1. LOAD EXISTING METHYLATION DATASET
# ============================================================================
print("\n[1/8] Loading methylation dataset...")

df_meth = pd.read_csv('data/processed/ML_dataset_methylation_features.csv.gz',
                      index_col=0, compression='gzip')

print(f"  ✓ Loaded: {df_meth.shape}")
print(f"  ✓ Cell lines: {df_meth.shape[0]:,}")
print(f"  ✓ Total columns: {df_meth.shape[1]:,}")

# Identify column types
metadata_cols = ['primary site', 'primary histology', 'cosmic_id']
cpg_cols = [col for col in df_meth.columns if col.startswith('cg')]

print(f"  ✓ Metadata columns: {len(metadata_cols)}")
print(f"  ✓ CpG features: {len(cpg_cols):,}")

# ============================================================================
# 2. LOAD GDSC DRUG RESPONSE DATA
# ============================================================================
print("\n[2/8] Loading GDSC drug response data...")
print("  (This may take 30-60 seconds for the 28MB Excel file)")

df_gdsc_raw = pd.read_excel('data/raw/GDSC1_fitted_dose_response_27Oct23.xlsx')

print(f"  ✓ Loaded: {df_gdsc_raw.shape}")
print(f"  ✓ Entries (drug-cell line pairs): {len(df_gdsc_raw):,}")
print(f"  ✓ Unique cell lines: {df_gdsc_raw['CELL_LINE_NAME'].nunique():,}")
print(f"  ✓ Unique drugs: {df_gdsc_raw['DRUG_NAME'].nunique()}")

# Check available columns
print(f"\n  Available GDSC columns:")
for col in df_gdsc_raw.columns[:10]:
    print(f"    - {col}")

# ============================================================================
# 3. MATCH CELL LINES BETWEEN DATASETS
# ============================================================================
print("\n[3/8] Matching cell lines between GSE68379 and GDSC...")

def normalize_cell_line_name(name):
    """Normalize cell line name for matching."""
    return str(name).upper().replace('-', '').replace('_', '').replace(' ', '').strip()

# Get cell line names
gse_cell_lines = set(df_meth.index)
gdsc_cell_lines = set(df_gdsc_raw['CELL_LINE_NAME'].unique())

print(f"  GSE68379 cell lines: {len(gse_cell_lines):,}")
print(f"  GDSC cell lines: {len(gdsc_cell_lines):,}")

# Find exact matches
exact_matches = gse_cell_lines & gdsc_cell_lines
print(f"  Exact matches: {len(exact_matches)}")

# Create normalized mappings
gse_normalized = {normalize_cell_line_name(name): name for name in gse_cell_lines}
gdsc_normalized = {normalize_cell_line_name(name): name for name in gdsc_cell_lines}

# Find matches using normalized names
normalized_matches = set(gse_normalized.keys()) & set(gdsc_normalized.keys())
print(f"  Matches after normalization: {len(normalized_matches)}")

# Create mapping dictionary: GDSC name -> GSE name
cell_line_mapping = {}
for norm_name in normalized_matches:
    gdsc_name = gdsc_normalized[norm_name]
    gse_name = gse_normalized[norm_name]
    cell_line_mapping[gdsc_name] = gse_name

print(f"  ✓ Created mapping for {len(cell_line_mapping)} cell lines")

# Show examples
print(f"\n  Example mappings (first 10):")
for i, (gdsc_name, gse_name) in enumerate(list(cell_line_mapping.items())[:10], 1):
    match_type = "=" if gdsc_name == gse_name else "→"
    print(f"    {i:2d}. {gdsc_name:20s} {match_type} {gse_name}")

# ============================================================================
# 4. FILTER GDSC DATA TO MATCHED CELL LINES
# ============================================================================
print("\n[4/8] Filtering GDSC data to matched cell lines...")

df_gdsc_filtered = df_gdsc_raw[
    df_gdsc_raw['CELL_LINE_NAME'].isin(cell_line_mapping.keys())
].copy()

print(f"  GDSC entries before: {len(df_gdsc_raw):,}")
print(f"  GDSC entries after: {len(df_gdsc_filtered):,}")
print(f"  ✓ Unique matched cell lines: {df_gdsc_filtered['CELL_LINE_NAME'].nunique()}")
print(f"  ✓ Unique drugs in matched data: {df_gdsc_filtered['DRUG_NAME'].nunique()}")

# Map GDSC cell line names to GSE names
df_gdsc_filtered['GSE_CELL_LINE'] = df_gdsc_filtered['CELL_LINE_NAME'].map(cell_line_mapping)

# ============================================================================
# 5. ANALYZE DRUG COVERAGE AND FILTER
# ============================================================================
print("\n[5/8] Analyzing drug coverage...")

drug_coverage = df_gdsc_filtered.groupby('DRUG_NAME').size().sort_values(ascending=False)

print(f"  Total drugs: {len(drug_coverage)}")
print(f"  Coverage range: {drug_coverage.min()} - {drug_coverage.max()} cell lines")
print(f"  Mean coverage: {drug_coverage.mean():.1f} cell lines")

# Filter to high-quality drugs (≥100 cell lines tested)
min_coverage = 100
high_coverage_drugs = drug_coverage[drug_coverage >= min_coverage]

print(f"\n  Filtering to drugs tested in ≥{min_coverage} cell lines...")
print(f"  ✓ High-quality drugs: {len(high_coverage_drugs)}")

# Show top drugs
print(f"\n  Top 15 drugs by coverage:")
for i, (drug, count) in enumerate(drug_coverage.head(15).items(), 1):
    print(f"    {i:2d}. {drug:40s} ({count} cell lines)")

# Filter dataset
high_quality_drugs_list = high_coverage_drugs.index.tolist()
df_gdsc_hq = df_gdsc_filtered[df_gdsc_filtered['DRUG_NAME'].isin(high_quality_drugs_list)].copy()

print(f"\n  Filtered GDSC data:")
print(f"  ✓ Entries: {len(df_gdsc_hq):,}")
print(f"  ✓ Cell lines: {df_gdsc_hq['GSE_CELL_LINE'].nunique()}")
print(f"  ✓ Drugs: {df_gdsc_hq['DRUG_NAME'].nunique()}")

# ============================================================================
# 6. CREATE DRUG RESPONSE MATRIX
# ============================================================================
print("\n[6/8] Creating drug response matrix...")

# Use LN_IC50 as the primary metric
# Lower LN_IC50 = more sensitive to drug
df_drug_response = df_gdsc_hq.pivot_table(
    index='GSE_CELL_LINE',
    columns='DRUG_NAME',
    values='LN_IC50',
    aggfunc='mean'  # Average if multiple measurements
)

print(f"  ✓ Drug response matrix: {df_drug_response.shape}")
print(f"  ✓ Cell lines (rows): {df_drug_response.shape[0]}")
print(f"  ✓ Drugs (columns): {df_drug_response.shape[1]}")

missing_pct = df_drug_response.isna().sum().sum() / df_drug_response.size * 100
print(f"  ✓ Missing data: {missing_pct:.2f}%")

# ============================================================================
# 7. COMBINE METHYLATION AND DRUG RESPONSE
# ============================================================================
print("\n[7/8] Combining methylation and drug response data...")

# Find common cell lines
meth_cell_lines = set(df_meth.index)
drug_cell_lines = set(df_drug_response.index)
common_cell_lines = sorted(list(meth_cell_lines & drug_cell_lines))

print(f"  Cell lines with methylation data: {len(meth_cell_lines)}")
print(f"  Cell lines with drug response data: {len(drug_cell_lines)}")
print(f"  ✓ Cell lines with BOTH: {len(common_cell_lines)}")

# Subset to common cell lines
df_meth_common = df_meth.loc[common_cell_lines].copy()
df_drug_common = df_drug_response.loc[common_cell_lines].copy()

print(f"\n  Methylation subset: {df_meth_common.shape}")
print(f"  Drug response subset: {df_drug_common.shape}")

# Combine datasets
# Structure: [metadata] [methylation features] [drug targets]
df_combined = pd.concat([
    df_meth_common[metadata_cols],      # Metadata
    df_meth_common[cpg_cols],           # Methylation features
    df_drug_common                       # Drug response targets
], axis=1)

print(f"\n  ✓ Combined dataset: {df_combined.shape}")
print(f"\n  Column structure:")
print(f"    1. Metadata: {len(metadata_cols)} columns")
print(f"    2. Methylation features: {len(cpg_cols):,} CpG sites")
print(f"    3. Drug response targets: {len(df_drug_common.columns)} drugs")
print(f"    Total: {df_combined.shape[1]:,} columns")

# ============================================================================
# 8. SAVE OUTPUTS
# ============================================================================
print("\n[8/8] Saving outputs...")

# Save combined dataset
output_file = Path('data/processed/ML_dataset_methylation_drug_response.csv.gz')
df_combined.to_csv(output_file, compression='gzip')
file_size_mb = output_file.stat().st_size / (1024 * 1024)
print(f"  ✓ Saved: {output_file.name} ({file_size_mb:.2f} MB)")

# Create column information file
drug_cols = df_drug_common.columns.tolist()
column_info = pd.DataFrame({
    'column_name': df_combined.columns,
    'column_type': (
        ['metadata'] * len(metadata_cols) +
        ['methylation_feature'] * len(cpg_cols) +
        ['drug_response_target'] * len(drug_cols)
    ),
    'data_type': df_combined.dtypes.values.astype(str),
    'missing_count': df_combined.isna().sum().values,
    'missing_pct': (df_combined.isna().sum() / len(df_combined) * 100).values
})

column_info_file = Path('data/processed/ML_dataset_column_info.csv')
column_info.to_csv(column_info_file, index=False)
print(f"  ✓ Saved: {column_info_file.name}")

# Create summary statistics
meth_missing_pct = df_combined[cpg_cols].isna().sum().sum() / (len(cpg_cols) * len(df_combined)) * 100
drug_missing_pct_mean = (df_combined[drug_cols].isna().sum() / len(df_combined) * 100).mean()

summary_stats = {
    'dataset': 'GSE68379 (Methylation) + GDSC1 (Drug Response)',
    'created': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'total_samples': len(df_combined),
    'total_features': len(cpg_cols),
    'total_drug_targets': len(drug_cols),
    'total_columns': df_combined.shape[1],
    'file_size_mb': round(file_size_mb, 2),
    'methylation_missing_pct': round(meth_missing_pct, 4),
    'drug_response_missing_pct_mean': round(drug_missing_pct_mean, 2),
    'primary_sites': df_combined['primary site'].nunique(),
    'histologies': df_combined['primary histology'].nunique()
}

summary_file = Path('data/processed/ML_dataset_summary.txt')
with open(summary_file, 'w') as f:
    f.write("ML-Ready Methylation + Drug Response Dataset Summary\n")
    f.write("=" * 60 + "\n\n")
    for key, value in summary_stats.items():
        f.write(f"{key:40s}: {value}\n")
    f.write("\n" + "=" * 60 + "\n\n")
    f.write("Dataset Structure:\n")
    f.write(f"  Rows (cell lines): {len(df_combined)}\n")
    f.write(f"  Columns:\n")
    f.write(f"    - Metadata: {len(metadata_cols)}\n")
    f.write(f"    - Methylation features: {len(cpg_cols):,} CpG sites\n")
    f.write(f"    - Drug response targets: {len(drug_cols)} drugs\n")

print(f"  ✓ Saved: {summary_file.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("INTEGRATION COMPLETE!")
print("=" * 80)

print(f"\n📊 DATASET SUMMARY:")
print(f"  • Cell lines: {len(df_combined):,}")
print(f"  • Methylation features: {len(cpg_cols):,} CpG sites")
print(f"  • Drug targets: {len(drug_cols)} drugs")
print(f"  • Total columns: {df_combined.shape[1]:,}")
print(f"  • File size: {file_size_mb:.2f} MB")

print(f"\n📁 OUTPUT FILES:")
print(f"  1. {output_file.name}")
print(f"  2. {column_info_file.name}")
print(f"  3. {summary_file.name}")

print(f"\n🎯 READY FOR MACHINE LEARNING!")

# ============================================================================
# DETAILED DATASET SUMMARY FOR USER
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED DATASET ANALYSIS")
print("=" * 80)

print("\n[A] METADATA COLUMNS (3)")
print("-" * 60)
for col in metadata_cols:
    nunique = df_combined[col].nunique()
    print(f"  {col:25s}: {nunique:4d} unique values")
    if nunique <= 15:
        top_vals = df_combined[col].value_counts().head(5)
        for val, count in top_vals.items():
            print(f"    - {str(val):40s}: {count:3d} samples")

print("\n[B] METHYLATION FEATURES (10,000 CpG sites)")
print("-" * 60)
print(f"  Column pattern: cg########")
print(f"  Data type: float64")
print(f"  Value range: [0.0, 1.0] (beta values)")
print(f"  Missing: {meth_missing_pct:.3f}%")
print(f"  Example CpG sites:")
for i, cpg in enumerate(cpg_cols[:5], 1):
    mean_val = df_combined[cpg].mean()
    std_val = df_combined[cpg].std()
    print(f"    {i}. {cpg}: mean={mean_val:.3f}, std={std_val:.3f}")

print("\n[C] DRUG RESPONSE TARGETS ({} drugs)".format(len(drug_cols)))
print("-" * 60)
print(f"  Data type: float64 (LN_IC50 values)")
print(f"  Interpretation: Lower = more sensitive to drug")
print(f"  Missing data per drug: {drug_missing_pct_mean:.1f}% (average)")

# Analyze missing data per drug
drug_missing = df_combined[drug_cols].isna().sum()
drug_missing_pct = drug_missing / len(df_combined) * 100

print(f"\n  Drug completeness distribution:")
print(f"    Drugs with <10% missing: {(drug_missing_pct < 10).sum()}")
print(f"    Drugs with <20% missing: {(drug_missing_pct < 20).sum()}")
print(f"    Drugs with <30% missing: {(drug_missing_pct < 30).sum()}")

print(f"\n  Top 20 drugs by data completeness:")
for i, (drug, missing_pct) in enumerate(drug_missing_pct.sort_values().head(20).items(), 1):
    n_samples = len(df_combined) - drug_missing[drug]
    print(f"    {i:2d}. {drug:40s}: {n_samples:3d} samples ({100-missing_pct:.1f}% complete)")

print("\n[D] SAMPLE CHARACTERISTICS")
print("-" * 60)
print("\nTissue distribution (primary site):")
tissue_dist = df_combined['primary site'].value_counts()
for tissue, count in tissue_dist.items():
    pct = count / len(df_combined) * 100
    print(f"  {tissue:45s}: {count:3d} ({pct:5.1f}%)")

print("\n[E] EXAMPLE USE CASES")
print("-" * 60)
print("  1. Single-drug response prediction:")
print("     X = methylation features (10,000 CpG sites)")
print("     y = specific drug LN_IC50 values")
print("     Model: RandomForest, XGBoost, Neural Network")
print()
print("  2. Multi-drug response prediction:")
print("     X = methylation features")
print("     y = multiple drug responses (multi-output)")
print()
print("  3. Drug sensitivity classification:")
print("     X = methylation features")
print("     y = binary (sensitive vs resistant, using IC50 threshold)")
print()
print("  4. Biomarker discovery:")
print("     Train model → extract feature importance")
print("     → identify CpG sites that predict drug response")

print("\n" + "=" * 80)
print("Integration script completed successfully!")
print("=" * 80)
