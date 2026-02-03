"""
View All Available Drugs in the Dataset
=========================================

This script shows how to extract and view all 375 drugs in the dataset,
along with their data completeness.

Usage:
    python examples/view_all_drugs.py

Output:
    - Prints all drugs to console
    - Saves drug list to 'available_drugs.csv'
"""

import pandas as pd
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("PHARMACOEPIGENOMICS DATASET - DRUG CATALOG")
print("=" * 80)

# Load dataset
print("\nLoading dataset...")
try:
    df = pd.read_csv('data/processed/ML_dataset_methylation_drug_response.csv.gz',
                     index_col=0, compression='gzip')
    print(f"✓ Loaded: {df.shape}")
except FileNotFoundError:
    print("ERROR: Dataset not found!")
    print("Expected location: data/processed/ML_dataset_methylation_drug_response.csv.gz")
    sys.exit(1)

# Identify column types
metadata_cols = ['primary site', 'primary histology', 'cosmic_id']
feature_cols = [col for col in df.columns if col.startswith('cg')]
drug_cols = [col for col in df.columns
             if col not in metadata_cols and not col.startswith('cg')]

print(f"\nDataset breakdown:")
print(f"  Samples: {len(df)}")
print(f"  Metadata columns: {len(metadata_cols)}")
print(f"  Methylation features: {len(feature_cols):,}")
print(f"  Drug targets: {len(drug_cols)}")

# Calculate completeness for each drug
print(f"\nCalculating drug completeness...")
drug_info = pd.DataFrame({
    'Drug': drug_cols,
    'Samples': df[drug_cols].notna().sum().values,
    'Percent_Complete': (df[drug_cols].notna().sum() / len(df) * 100).values
}).sort_values('Samples', ascending=False)

# Add rank
drug_info.insert(0, 'Rank', range(1, len(drug_info) + 1))

# Summary statistics
print(f"\n" + "=" * 80)
print("DRUG COVERAGE SUMMARY")
print("=" * 80)
print(f"Total drugs: {len(drug_info)}")
print(f"Average samples per drug: {drug_info['Samples'].mean():.1f}")
print(f"Median samples per drug: {drug_info['Samples'].median():.0f}")
print(f"Min samples: {drug_info['Samples'].min()}")
print(f"Max samples: {drug_info['Samples'].max()}")

# Coverage tiers
high_coverage = (drug_info['Percent_Complete'] >= 95).sum()
medium_coverage = ((drug_info['Percent_Complete'] >= 80) &
                   (drug_info['Percent_Complete'] < 95)).sum()
low_coverage = (drug_info['Percent_Complete'] < 80).sum()

print(f"\nDrug coverage tiers:")
print(f"  High (≥95%):   {high_coverage} drugs")
print(f"  Medium (80-95%): {medium_coverage} drugs")
print(f"  Low (<80%):    {low_coverage} drugs")

# Show top 20
print(f"\n" + "=" * 80)
print("TOP 20 DRUGS BY DATA COMPLETENESS")
print("=" * 80)
print(drug_info.head(20).to_string(index=False))

# Show bottom 20
print(f"\n" + "=" * 80)
print("BOTTOM 20 DRUGS BY DATA COMPLETENESS")
print("=" * 80)
print(drug_info.tail(20).to_string(index=False))

# Save to CSV
output_file = 'available_drugs.csv'
drug_info.to_csv(output_file, index=False)
print(f"\n" + "=" * 80)
print(f"✓ Full drug list saved to: {output_file}")
print("=" * 80)

# Optional: Show drugs by class/mechanism (if you know them)
print(f"\nTo view all {len(drug_info)} drugs, open: {output_file}")

# Print example of how to use specific drugs
print(f"\n" + "=" * 80)
print("EXAMPLE: How to Use This Information")
print("=" * 80)
print("""
# Load the drug list
drug_info = pd.read_csv('available_drugs.csv')

# Get drugs with >95% completeness
high_quality = drug_info[drug_info['Percent_Complete'] > 95]
print(f"High quality drugs: {len(high_quality)}")

# Select specific drugs for modeling
selected_drugs = high_quality.head(10)['Drug'].tolist()

# Train models on these drugs
for drug in selected_drugs:
    y = df[drug].dropna()
    X_subset = X.loc[y.index]
    # ... train model ...
""")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
