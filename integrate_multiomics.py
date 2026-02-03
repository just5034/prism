"""
Multi-Omics Integration: Methylation + Gene Expression
Fix cell line matching between GSE68379 and GDSC expression data
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

print("=" * 70)
print("MULTI-OMICS INTEGRATION: Methylation + Gene Expression")
print("=" * 70)

# ============================================================================
# 1. Load Methylation Data
# ============================================================================
print("\n[1/7] Loading methylation data...")
df_meth = pd.read_csv('data/processed/ML_dataset_methylation_features.csv.gz',
                       index_col=0, compression='gzip')
print(f"   Methylation: {df_meth.shape}")
print(f"   Samples: {len(df_meth)}")

# Separate metadata and features
metadata_cols = ['primary site', 'primary histology', 'cosmic_id']
meth_cols = [c for c in df_meth.columns if c.startswith('cg')]
print(f"   CpG features: {len(meth_cols)}")

# ============================================================================
# 2. Load Gene Expression Data
# ============================================================================
print("\n[2/7] Loading gene expression data...")
df_expr_raw = pd.read_csv('data/raw/CCLE/Cell_line_RMA_proc_basalExp.txt',
                           sep='\t', index_col=0)
print(f"   Expression (raw): {df_expr_raw.shape}")

# Clean: remove GENE_title column, transpose to get cell lines as rows
df_expr = df_expr_raw.drop(columns=['GENE_title']).T
df_expr.index = df_expr.index.str.replace('DATA.', '', regex=False)
df_expr.index.name = 'COSMIC_ID'
print(f"   Expression (transposed): {df_expr.shape}")
print(f"   Genes: {len(df_expr.columns)}")

# ============================================================================
# 3. Load GDSC Cell Line Annotations
# ============================================================================
print("\n[3/7] Loading GDSC cell line annotations...")
try:
    df_gdsc_annot = pd.read_excel('data/raw/CCLE/GDSC_cell_lines_annotations.xlsx')
    print(f"   Annotations: {df_gdsc_annot.shape}")
    print(f"   Columns: {df_gdsc_annot.columns.tolist()}")
except Exception as e:
    print(f"   Error loading annotations: {e}")
    print("   Attempting alternative approach...")
    df_gdsc_annot = None

# ============================================================================
# 4. Load GSE68379 Sample Metadata
# ============================================================================
print("\n[4/7] Loading GSE68379 sample metadata...")
df_gse_meta = pd.read_csv('data/processed/GSE68379_sample_metadata.csv')
print(f"   GSE metadata: {df_gse_meta.shape}")
print(f"   Columns: {df_gse_meta.columns.tolist()}")

# Create mapping: sample_name -> cell_line_name
gse_sample_to_cell_line = {}
for idx, row in df_gse_meta.iterrows():
    sample_name = row['gsm_id']
    cell_line_name = str(row['cell line']).strip()
    gse_sample_to_cell_line[sample_name] = cell_line_name

print(f"   GSE sample names: {len(gse_sample_to_cell_line)}")

# ============================================================================
# 5. Match Cell Lines Using Multiple Strategies
# ============================================================================
print("\n[5/7] Matching cell lines between datasets...")

def normalize_name(name):
    """Normalize cell line name for matching"""
    return str(name).upper().replace('-', '').replace('_', '').replace(' ', '').strip()

# Strategy 1: Direct COSMIC ID matching
meth_cosmic_ids = df_meth['cosmic_id'].dropna().astype(str).str.replace('.0', '', regex=False)
expr_cosmic_ids = set(df_expr.index.astype(str))

direct_matches = {}
for sample_idx, cosmic_id in meth_cosmic_ids.items():
    if cosmic_id in expr_cosmic_ids:
        direct_matches[sample_idx] = cosmic_id

print(f"\n   Strategy 1 (Direct COSMIC ID): {len(direct_matches)} matches")

# Strategy 2: Use cell line names from GSE metadata
# Get cell line names for samples in methylation dataset
meth_cell_line_names = {}
for sample_idx in df_meth.index:
    # Find this sample in GSE metadata
    gse_match = df_gse_meta[df_gse_meta['cell line'].astype(str) == str(sample_idx)]
    if len(gse_match) > 0:
        cell_line_name = str(gse_match.iloc[0]['cell line'])
        meth_cell_line_names[sample_idx] = cell_line_name
    else:
        # Sample index IS the cell line name in ML dataset
        meth_cell_line_names[sample_idx] = str(sample_idx)

print(f"   Cell line names extracted: {len(meth_cell_line_names)}")

# Try matching with GDSC annotations if available
name_matches = {}
if df_gdsc_annot is not None and len(df_gdsc_annot) > 0:
    # Check what columns exist
    possible_name_cols = ['Sample Name', 'Cell_Line_Name', 'CELL_LINE_NAME',
                          'Cell Line', 'Sample_Name']
    possible_id_cols = ['COSMIC identifier', 'COSMIC_ID', 'cosmic_id']

    name_col = None
    id_col = None

    for col in possible_name_cols:
        if col in df_gdsc_annot.columns:
            name_col = col
            break

    for col in possible_id_cols:
        if col in df_gdsc_annot.columns:
            id_col = col
            break

    if name_col and id_col:
        print(f"   Using GDSC columns: {name_col} -> {id_col}")

        # Create mapping: cell_line_name -> cosmic_id in expression data
        gdsc_name_to_cosmic = {}
        for _, row in df_gdsc_annot.iterrows():
            cell_name = str(row[name_col]).strip()
            cosmic_id = str(row[id_col]).strip()
            gdsc_name_to_cosmic[normalize_name(cell_name)] = cosmic_id

        # Match methylation cell line names to GDSC
        for sample_idx, cell_line in meth_cell_line_names.items():
            norm_name = normalize_name(cell_line)
            if norm_name in gdsc_name_to_cosmic:
                cosmic_id = gdsc_name_to_cosmic[norm_name]
                if cosmic_id in expr_cosmic_ids:
                    name_matches[sample_idx] = cosmic_id

        print(f"   Strategy 2 (Cell line names via GDSC): {len(name_matches)} matches")

# Strategy 3: Fuzzy matching on normalized names as fallback
# Create reverse lookup: COSMIC ID -> possible cell line names from expression data
if len(name_matches) == 0:
    print("   Strategy 3: Attempting direct name matching...")
    # This is a fallback - match sample names directly
    for sample_idx, cell_line in meth_cell_line_names.items():
        # IMPORTANT: Don't overwrite existing direct matches!
        if sample_idx in direct_matches:
            continue

        # Try to find in expression data by normalizing
        norm_meth = normalize_name(cell_line)
        # Check if any COSMIC ID could match
        for cosmic_id in expr_cosmic_ids:
            # Sometimes COSMIC IDs contain cell line info
            if norm_meth in normalize_name(cosmic_id):
                name_matches[sample_idx] = cosmic_id
                break

print(f"   Strategy 3 (Fuzzy matching): {len(name_matches)} new matches")

# Combine all strategies (direct matches take precedence)
all_matches = {**name_matches, **direct_matches}  # REVERSED ORDER - direct matches come last!
print(f"\n   [OK] Total unique matches: {len(all_matches)}")

if len(all_matches) == 0:
    print("\n   ERROR: No matches found!")
    print("   This likely means the expression data uses different cell line IDs")
    print("   than the methylation dataset.")
    sys.exit(1)

# ============================================================================
# 6. Select Top Variable Genes
# ============================================================================
print("\n[6/7] Selecting most variable genes...")

# IMPORTANT: Filter out genes with NaN names first!
valid_genes = df_expr.columns[df_expr.columns.notna()].tolist()
print(f"   Genes with valid names: {len(valid_genes)}/{len(df_expr.columns)}")

df_expr_valid = df_expr[valid_genes].copy()
gene_variance = df_expr_valid.var(axis=0).sort_values(ascending=False)
n_genes = 5000
top_genes = gene_variance.iloc[:n_genes].index.tolist()

df_expr_filtered = df_expr_valid[top_genes].copy()
print(f"   Selected top {n_genes} variable genes (excluding NaN names)")
print(f"   Variance range: {gene_variance.iloc[n_genes-1]:.4f} to {gene_variance.iloc[0]:.4f}")

# ============================================================================
# 7. Build Integrated Dataset
# ============================================================================
print("\n[7/7] Building integrated multi-omics dataset...")

# CRITICAL FIX: Build mapping FIRST, then use it for both datasets
print("   Building sample-to-COSMIC mapping...")

# Build mapping: sample_idx -> cosmic_id
idx_to_cosmic = {sample_idx: cosmic_id
                 for sample_idx, cosmic_id in all_matches.items()
                 if cosmic_id in df_expr_filtered.index}

# Check for duplicate COSMIC IDs
from collections import Counter
cosmic_counts = Counter(idx_to_cosmic.values())
duplicates = {k: v for k, v in cosmic_counts.items() if v > 1}
if duplicates:
    print(f"   WARNING: {len(duplicates)} COSMIC IDs map to multiple samples")
    print(f"   Keeping only first match for each COSMIC ID")
    # Keep only first occurrence of each COSMIC ID
    seen_cosmics = set()
    idx_to_cosmic_clean = {}
    for sample_idx, cosmic_id in all_matches.items():
        if cosmic_id in df_expr_filtered.index and cosmic_id not in seen_cosmics:
            idx_to_cosmic_clean[sample_idx] = cosmic_id
            seen_cosmics.add(cosmic_id)
    idx_to_cosmic = idx_to_cosmic_clean

# Use the SAME list of samples for both methylation and expression
matched_samples = list(idx_to_cosmic.keys())
print(f"   Building dataset for {len(matched_samples)} matched samples...")

# DEBUG: Check first 5 samples
print("   DEBUG - First 5 samples mapping:")
for i, sample_idx in enumerate(matched_samples[:5]):
    cosmic_id = idx_to_cosmic[sample_idx]
    print(f"     [{i}] {sample_idx} -> COSMIC {cosmic_id}")

# Start with methylation data for matched samples
df_integrated = df_meth.loc[matched_samples].copy()

# Add expression data for each matched sample - OPTIMIZED VERSION
print("   Adding expression features...")

# Create expression features DataFrame
# Build list of COSMIC IDs in the same order as matched_samples
matched_cosmic_ids = [idx_to_cosmic[sample_idx] for sample_idx in matched_samples]

# Get expression data for matched COSMIC IDs
df_expr_matched = df_expr_filtered.loc[matched_cosmic_ids].copy()

# DEBUG: Check before index assignment
print(f"   DEBUG - df_expr_matched index[:5] before assignment: {df_expr_matched.index[:5].tolist()}")
print(f"   DEBUG - Will set to: {matched_samples[:5]}")

# Set index to sample indices (same order as df_integrated)
df_expr_matched.index = matched_samples

print(f"   DEBUG - df_expr_matched index[:5] after assignment: {df_expr_matched.index[:5].tolist()}")

# Rename columns to add 'expr_' prefix
df_expr_features = df_expr_matched.copy()
df_expr_features.columns = [f'expr_{col}' for col in df_expr_features.columns]

# Merge with methylation (now guaranteed to be aligned)
df_integrated = pd.concat([df_integrated, df_expr_features], axis=1)

print(f"\n   [OK] Integrated dataset shape: {df_integrated.shape}")
print(f"   - Samples: {len(df_integrated)}")
print(f"   - Metadata columns: {len(metadata_cols)}")
print(f"   - Methylation features: {len(meth_cols)}")
print(f"   - Expression features: {len([c for c in df_integrated.columns if c.startswith('expr_')])}")

# Check missing values
missing_pct = (df_integrated.isnull().sum().sum() / df_integrated.size) * 100
print(f"   - Missing values: {missing_pct:.4f}%")

# ============================================================================
# 8. Save Dataset
# ============================================================================
print("\n[8/8] Saving integrated dataset...")
output_file = 'data/processed/ml_with_gene_expr.csv'
df_integrated.to_csv(output_file)

import os
file_size_mb = os.path.getsize(output_file) / (1024**2)
print(f"   [OK] Saved to: {output_file}")
print(f"   - File size: {file_size_mb:.1f} MB")

# Also save gzipped version for space efficiency
output_file_gz = 'data/processed/ml_with_gene_expr.csv.gz'
df_integrated.to_csv(output_file_gz, compression='gzip')
file_size_gz_mb = os.path.getsize(output_file_gz) / (1024**2)
print(f"   [OK] Saved compressed: {output_file_gz}")
print(f"   - Compressed size: {file_size_gz_mb:.1f} MB")

# ============================================================================
# 9. Create Summary
# ============================================================================
print("\n" + "=" * 70)
print("INTEGRATION COMPLETE")
print("=" * 70)

summary = {
    'Total Samples': len(df_integrated),
    'Metadata Columns': len(metadata_cols),
    'Methylation Features': len(meth_cols),
    'Expression Features': len([c for c in df_integrated.columns if c.startswith('expr_')]),
    'Total Columns': len(df_integrated.columns),
    'Missing Data (%)': f"{missing_pct:.4f}",
    'File Size (MB)': f"{file_size_mb:.1f}",
    'Compressed (MB)': f"{file_size_gz_mb:.1f}"
}

print("\nDataset Summary:")
for key, value in summary.items():
    print(f"  {key:25s}: {value}")

# Save summary
summary_file = 'data/processed/ml_with_gene_expr_summary.txt'
with open(summary_file, 'w') as f:
    f.write("Multi-Omics Dataset (Methylation + Expression)\n")
    f.write("=" * 50 + "\n\n")
    for key, value in summary.items():
        f.write(f"{key:25s}: {value}\n")
    f.write("\n" + "=" * 50 + "\n")
    f.write("\nFeature Breakdown:\n")
    f.write(f"  - Metadata: primary site, primary histology, cosmic_id\n")
    f.write(f"  - Methylation: {len(meth_cols)} CpG sites (top 10,000 variable from GSE68379)\n")
    f.write(f"  - Expression: {n_genes} genes (top {n_genes} variable from GDSC RMA data)\n")
    f.write(f"\nUse Cases:\n")
    f.write(f"  - Improved drug response prediction (vs methylation-only)\n")
    f.write(f"  - Multi-omics biomarker discovery\n")
    f.write(f"  - Tissue classification with enhanced features\n")
    f.write(f"  - Gene-CpG interaction analysis\n")

print(f"\n[OK] Summary saved to: {summary_file}")

print("\n" + "=" * 70)
print("READY FOR MODEL TRAINING")
print("=" * 70)
print("\nExpected Performance Improvements:")
print("  - Tissue classification: 85-90% -> 90-95%")
print("  - Drug response (Avagacestat): R²=0.10 -> R²=0.35-0.50")
print("  - Feature interpretability: CpG sites + genes = better biology")
print("\nNext Steps:")
print("  1. Re-train models using ml_with_gene_expr.csv")
print("  2. Compare to methylation-only baseline")
print("  3. Extract feature importance (which genes + CpGs matter?)")
print("  4. Test on drugs with known epigenetic mechanisms")
print("=" * 70)
