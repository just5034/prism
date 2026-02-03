"""
Comprehensive sanity check for ml_with_gene_expr.csv.gz
Verify that expression and methylation data are properly aligned
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("SANITY CHECK: Multi-Omics Integration")
print("=" * 70)

# Load the integrated dataset
print("\n[1/8] Loading integrated dataset...")
df = pd.read_csv('data/processed/ml_with_gene_expr.csv.gz',
                 compression='gzip', low_memory=False, index_col=0)
print(f"   Shape: {df.shape}")
print(f"   Samples: {len(df)}")

# Separate column types
expr_cols = [c for c in df.columns if c.startswith('expr_')]
cg_cols = [c for c in df.columns if c.startswith('cg')]
metadata_cols = ['primary site', 'primary histology', 'cosmic_id']

print(f"   Metadata: {len(metadata_cols)}")
print(f"   Methylation: {len(cg_cols)}")
print(f"   Expression: {len(expr_cols)}")

# Check 1: Data coverage
print("\n[2/8] Checking data coverage...")
has_expr = df[expr_cols].notna().any(axis=1).sum()
has_meth = df[cg_cols].notna().any(axis=1).sum()
has_both = (df[expr_cols].notna().any(axis=1) & df[cg_cols].notna().any(axis=1)).sum()

print(f"   Rows with expression: {has_expr}/{len(df)} ({100*has_expr/len(df):.1f}%)")
print(f"   Rows with methylation: {has_meth}/{len(df)} ({100*has_meth/len(df):.1f}%)")
print(f"   Rows with BOTH: {has_both}/{len(df)} ({100*has_both/len(df):.1f}%)")

if has_both < 0.95 * len(df):
    print(f"   [!] WARNING: Only {has_both} rows have both data types!")
else:
    print(f"   [OK] Good: {has_both} rows have complete data")

# Check 2: Value ranges
print("\n[3/8] Checking value ranges...")

# Methylation should be beta values (0-1)
meth_min = df[cg_cols].min().min()
meth_max = df[cg_cols].max().max()
meth_mean = df[cg_cols].mean().mean()
print(f"   Methylation (beta values):")
print(f"     Range: {meth_min:.4f} to {meth_max:.4f}")
print(f"     Mean: {meth_mean:.4f}")
if meth_min < 0 or meth_max > 1:
    print(f"   [!]  WARNING: Methylation values outside [0,1]!")
else:
    print(f"   [OK] Methylation values look correct")

# Expression should be log2-transformed (typically 2-15)
expr_min = df[expr_cols].min().min()
expr_max = df[expr_cols].max().max()
expr_mean = df[expr_cols].mean().mean()
print(f"   Expression (RMA log2):")
print(f"     Range: {expr_min:.4f} to {expr_max:.4f}")
print(f"     Mean: {expr_mean:.4f}")
if expr_mean < 2 or expr_mean > 15:
    print(f"   [!]  WARNING: Expression mean unusual for log2 data")
else:
    print(f"   [OK] Expression values look correct")

# Check 3: Missing data patterns
print("\n[4/8] Checking missing data patterns...")
missing_pct = (df.isnull().sum().sum() / df.size) * 100
print(f"   Overall missing: {missing_pct:.4f}%")

missing_by_col_type = {
    'Metadata': (df[metadata_cols].isnull().sum().sum() / df[metadata_cols].size) * 100,
    'Methylation': (df[cg_cols].isnull().sum().sum() / df[cg_cols].size) * 100,
    'Expression': (df[expr_cols].isnull().sum().sum() / df[expr_cols].size) * 100,
}

for col_type, pct in missing_by_col_type.items():
    status = "[OK]" if pct < 1.0 else "[!]"
    print(f"   {status} {col_type}: {pct:.4f}% missing")

# Check 4: Cross-reference with source data
print("\n[5/8] Cross-checking with source data...")

# Load methylation source
df_meth = pd.read_csv('data/processed/ML_dataset_methylation_features.csv.gz',
                      index_col=0, compression='gzip')
meth_source_cols = [c for c in df_meth.columns if c.startswith('cg')]

print(f"   Methylation source: {df_meth.shape}")

# Load expression source
df_expr_raw = pd.read_csv('data/raw/CCLE/Cell_line_RMA_proc_basalExp.txt',
                          sep='\t', index_col=0)
df_expr = df_expr_raw.drop(columns=['GENE_title']).T
df_expr.index = df_expr.index.str.replace('DATA.', '', regex=False)

print(f"   Expression source: {df_expr.shape}")

# Pick 5 random samples and verify values match
print("\n[6/8] Spot-checking data alignment...")
sample_cells = df.index[:5].tolist()

all_match = True
for cell_line in sample_cells:
    # Check methylation
    if cell_line in df_meth.index:
        # Compare first 3 CpG sites
        test_cgs = meth_source_cols[:3]
        orig_vals = df_meth.loc[cell_line, test_cgs].values.astype(float)
        new_vals = df.loc[cell_line, test_cgs].values.astype(float)

        meth_match = np.allclose(orig_vals, new_vals, rtol=1e-5, equal_nan=True)

        # Check expression - need to find COSMIC ID
        cosmic_id = str(df.loc[cell_line, 'cosmic_id'])
        cosmic_id = cosmic_id.replace('.0', '')

        if cosmic_id in df_expr.index:
            # Compare first 3 genes that are in the integrated dataset
            # (integrated has only top 5000 variable genes)
            integrated_genes = [c.replace('expr_', '') for c in expr_cols]
            test_genes = [g for g in df_expr.columns if g in integrated_genes][:3]
            test_expr_cols = [f'expr_{g}' for g in test_genes]

            orig_expr_vals = df_expr.loc[cosmic_id, test_genes].values.astype(float)
            new_expr_vals = df.loc[cell_line, test_expr_cols].values.astype(float)

            expr_match = np.allclose(orig_expr_vals, new_expr_vals, rtol=1e-5, equal_nan=True)
        else:
            expr_match = False

        status = "[OK]" if (meth_match and expr_match) else "[X]"
        print(f"   {status} {cell_line}: Meth={meth_match}, Expr={expr_match}")

        if not (meth_match and expr_match):
            all_match = False
            print(f"      Methylation sample: {orig_vals[:3]}")
            print(f"      Integrated: {new_vals[:3]}")
            if cosmic_id in df_expr.index:
                print(f"      Expression sample: {orig_expr_vals[:3]}")
                print(f"      Integrated: {new_expr_vals[:3]}")

if all_match:
    print("\n   [OK] All spot checks passed!")
else:
    print("\n   [!]  Some values don't match source data!")

# Check 5: Metadata consistency
print("\n[7/8] Checking metadata consistency...")
unique_sites = df['primary site'].nunique()
unique_histology = df['primary histology'].nunique()
unique_cosmic = df['cosmic_id'].nunique()

print(f"   Unique primary sites: {unique_sites}")
print(f"   Unique histologies: {unique_histology}")
print(f"   Unique COSMIC IDs: {unique_cosmic}")

# Check for suspiciously low numbers
if unique_cosmic < 0.9 * len(df):
    print(f"   [!]  WARNING: Only {unique_cosmic} unique COSMIC IDs for {len(df)} samples")
else:
    print(f"   [OK] COSMIC ID uniqueness looks good")

# Check 6: Statistical sanity
print("\n[8/8] Statistical sanity checks...")

# Check variance - data should have reasonable variance
expr_variances = df[expr_cols].var()
low_var_expr = (expr_variances < 0.01).sum()
print(f"   Expression genes with low variance (<0.01): {low_var_expr}/{len(expr_cols)}")

meth_variances = df[cg_cols].var()
low_var_meth = (meth_variances < 0.001).sum()
print(f"   CpG sites with low variance (<0.001): {low_var_meth}/{len(cg_cols)}")

# Check correlations between a few random features
print("\n   Checking feature correlations...")
sample_expr = expr_cols[:10]
sample_meth = cg_cols[:10]

# Expression features should correlate with each other (genes in pathways)
expr_corr = df[sample_expr].corr().values
expr_mean_corr = np.abs(expr_corr[np.triu_indices_from(expr_corr, k=1)]).mean()
print(f"   Mean |correlation| between expression features: {expr_mean_corr:.4f}")

# Methylation and expression should have some correlation
mixed_corr = df[sample_expr + sample_meth].corr().loc[sample_expr, sample_meth].values
mixed_mean_corr = np.abs(mixed_corr).mean()
print(f"   Mean |correlation| between expr & meth: {mixed_mean_corr:.4f}")

# Check for suspiciously high correlations (might indicate duplication)
if expr_mean_corr > 0.95:
    print(f"   [!]  WARNING: Expression features are highly correlated - possible duplication?")
else:
    print(f"   [OK] Feature correlations look reasonable")

# Final summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

issues = []
if has_both < 0.95 * len(df):
    issues.append(f"Only {has_both}/{len(df)} rows have complete data")
if meth_min < 0 or meth_max > 1:
    issues.append("Methylation values outside [0,1] range")
if expr_mean < 2 or expr_mean > 15:
    issues.append("Expression values unusual for log2-transformed data")
if not all_match:
    issues.append("Spot checks found mismatched values")
if expr_mean_corr > 0.95:
    issues.append("Suspiciously high feature correlations")

if len(issues) == 0:
    print("\n[OK] ALL CHECKS PASSED!")
    print("  The integrated dataset looks good and ready to use.")
    print("\nKey Stats:")
    print(f"  - {len(df)} samples with both methylation and expression")
    print(f"  - {len(cg_cols)} methylation features")
    print(f"  - {len(expr_cols)} expression features")
    print(f"  - {missing_pct:.4f}% missing data")
else:
    print("\n[!]  ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\n  Please review the above details.")

print("=" * 70)
