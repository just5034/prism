"""
Simple script to directly download processed GSE270494 methylation data.
This downloads only the processed beta-value CSV files, not raw IDATs.
"""

import requests
from pathlib import Path
from tqdm import tqdm

# Create data directory
DATA_DIR = Path("data/raw/GSE270494")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Direct URLs for processed beta-value files
FILES_TO_DOWNLOAD = {
    "GSE270494_Noguera-Castells_Average_Beta_Homo_Sapiens.csv.gz":
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270494/suppl/GSE270494_Noguera-Castells_Average_Beta_Homo_Sapiens.csv.gz",
    "GSE270494_Noguera-Castells_Average_Beta_Mus_Musculus.csv.gz":
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270494/suppl/GSE270494_Noguera-Castells_Average_Beta_Mus_Musculus.csv.gz"
}

def download_file(url, dest_path):
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"  ✓ {dest_path.name} already exists")
        return

    print(f"  Downloading {dest_path.name}...")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"  ✓ Successfully downloaded {dest_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial download
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("Downloading GSE270494 Processed Methylation Data")
    print("=" * 80)
    print(f"Destination: {DATA_DIR.absolute()}\n")

    success_count = 0
    for filename, url in FILES_TO_DOWNLOAD.items():
        dest_path = DATA_DIR / filename
        if download_file(url, dest_path):
            success_count += 1

    print(f"\n{'=' * 80}")
    print(f"Download complete: {success_count}/{len(FILES_TO_DOWNLOAD)} files")
    print(f"{'=' * 80}")
