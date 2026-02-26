"""Check SMILES coverage against the drugs we actually model."""

import csv
import gzip
import json
from pathlib import Path

data_dir = Path("data")

# Load SMILES cache
cache_path = data_dir / "drug_smiles.cache.json"
with open(cache_path) as f:
    cache = json.load(f)

# Load drug response dataset header
dr_path = data_dir / "processed" / "ML_dataset_methylation_drug_response.csv.gz"
with gzip.open(dr_path, "rt") as f:
    header = f.readline().strip().split(",")

# Drug columns: everything after metadata (3) + CpG features (10000)
drug_cols = [c for c in header[3:] if not c.startswith("cg")]

found = [d for d in drug_cols if cache.get(d)]
missing = [d for d in drug_cols if not cache.get(d)]

print(f"Modeled drugs:  {len(drug_cols)}")
print(f"With SMILES:    {len(found)} ({len(found)/len(drug_cols)*100:.0f}%)")
print(f"Missing SMILES: {len(missing)}")
print()

if missing:
    print("Missing drugs:")
    for d in sorted(missing):
        print(f"  {d}")
