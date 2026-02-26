"""Check SMILES coverage against GDSC drugs.

Uses the downloaded compounds CSV and drug response XLSX to determine
which drugs we actually model, then checks SMILES cache coverage.
"""

import csv
import json
from pathlib import Path

data_dir = Path("data")

# Load SMILES cache
cache_path = data_dir / "drug_smiles.cache.json"
with open(cache_path) as f:
    cache = json.load(f)

# --- All compounds from GDSC compounds CSV ---
compounds_path = data_dir / "raw" / "GDSC" / "screened_compounds_rel_8.5.csv"
all_drugs = set()
with open(compounds_path, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    # Find drug name column
    name_col = None
    for h in headers:
        if "drug_name" in h.lower() or h.lower() == "name":
            name_col = h
            break
    if name_col is None:
        name_col = headers[0]
    for row in reader:
        name = row.get(name_col, "").strip()
        if name:
            all_drugs.add(name)

print(f"=== All GDSC compounds ===")
print(f"Total:          {len(all_drugs)}")
found_all = [d for d in all_drugs if cache.get(d)]
print(f"With SMILES:    {len(found_all)} ({len(found_all)/len(all_drugs)*100:.0f}%)")
print(f"Missing:        {len(all_drugs) - len(found_all)}")
print()

# --- Try to identify drugs in our actual GDSC1 drug response data ---
# Read drug names from the XLSX via the GDSC1 dose response file
# Since we can't read XLSX without openpyxl, use a different approach:
# read the drug names that appear in GDSC1 from the compounds CSV
# (DATASET column or similar)
print(f"=== Compounds CSV columns ===")
print(f"Headers: {headers}")
print()

# Show a few sample rows to understand structure
with open(compounds_path, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    print("First 3 rows:")
    for i, row in enumerate(reader):
        if i >= 3:
            break
        print(f"  {dict(row)}")
print()

# --- Report missing drugs ---
missing = sorted(d for d in all_drugs if not cache.get(d))
print(f"=== Missing drugs ({len(missing)}) ===")
for d in missing:
    print(f"  {d}")
