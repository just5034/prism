"""
Download All Raw Data for Cancer Drug Response Prediction
=========================================================

Single script to download all public data sources needed for the GNN pipeline:
  - GSE68379 methylation (485K CpGs, 3.9 GB)
  - GDSC gene expression (~19K genes, 137 MB)
  - GDSC1 drug response IC50 (28 MB)
  - GDSC compound annotations
  - GDSC cell line annotations
  - STRING v12 human PPI network (79 MB)
  - STRING v12 protein info (1.9 MB)
  - Illumina 450K manifest (192 MB)
  - Drug SMILES from PubChem API

Usage:
    python download_all_data.py                          # download everything
    python download_all_data.py --data-dir /scratch/x    # custom data root
    python download_all_data.py --skip-smiles            # skip PubChem API calls
    python download_all_data.py --only methylation expression  # selective download

Total download: ~4.4 GB
"""

import argparse
import csv
import gzip
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

# =============================================================================
# CONSTANTS
# =============================================================================

DOWNLOADS = {
    "methylation": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68379/suppl/GSE68379_Matrix.processed.txt.gz",
        "dest": "raw/GSE68379/GSE68379_Matrix.processed.txt.gz",
        "min_size_mb": 3500,
        "description": "GSE68379 methylation matrix (485K CpGs, ~3.9 GB)",
    },
    "expression": {
        "url": "https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip",
        "dest": "raw/GDSC/Cell_line_RMA_proc_basalExp.txt.zip",
        "min_size_mb": 100,
        "description": "GDSC gene expression RMA (~19K genes, 137 MB zip)",
        "extract": {
            "member": "Cell_line_RMA_proc_basalExp.txt",
            "dest": "raw/GDSC/Cell_line_RMA_proc_basalExp.txt",
        },
    },
    "drug_response": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx",
        "dest": "raw/GDSC/GDSC1_fitted_dose_response_27Oct23.xlsx",
        "min_size_mb": 20,
        "description": "GDSC1 fitted dose response IC50 (28 MB)",
    },
    "compounds": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/screened_compounds_rel_8.5.csv",
        "dest": "raw/GDSC/screened_compounds_rel_8.5.csv",
        "min_size_mb": 0.001,
        "description": "GDSC screened compound annotations",
    },
    "cell_lines": {
        "url": "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx",
        "dest": "raw/GDSC/Cell_Lines_Details.xlsx",
        "min_size_mb": 0.01,
        "description": "GDSC cell line annotations",
    },
    "string_ppi": {
        "url": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        "dest": "raw/STRING/9606.protein.links.v12.0.txt.gz",
        "min_size_mb": 50,
        "description": "STRING v12 human PPI network (79 MB)",
    },
    "string_info": {
        "url": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        "dest": "raw/STRING/9606.protein.info.v12.0.txt.gz",
        "min_size_mb": 1,
        "description": "STRING v12 human protein info (1.9 MB)",
    },
    "illumina_manifest": {
        "url": "https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv",
        "dest": "raw/Illumina/humanmethylation450_15017482_v1-2.csv",
        "min_size_mb": 100,
        "description": "Illumina 450K array manifest (192 MB)",
    },
}

# PubChem API config
PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_RATE_LIMIT_DELAY = 0.25  # seconds between requests
PUBCHEM_TIMEOUT = 15  # seconds per request
PUBCHEM_CACHE_SAVE_INTERVAL = 25  # save cache every N drugs

logger = logging.getLogger("download_all_data")


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: Path) -> None:
    """Configure dual logging to file (DEBUG) and stdout (INFO).

    Parameters
    ----------
    log_file : Path
        Path to log file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)

    # File handler - DEBUG level
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler - INFO level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download all raw data for cancer drug response prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--skip-smiles",
        action="store_true",
        help="Skip PubChem SMILES download",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(DOWNLOADS.keys()) + ["smiles"],
        help="Only download specified sources",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-download validation",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification (use if behind proxy)",
    )
    return parser.parse_args()


# =============================================================================
# DISK SPACE CHECK
# =============================================================================

def check_disk_space(data_dir: Path, required_gb: float = 6.0) -> None:
    """Warn if disk space is insufficient for the download.

    Parameters
    ----------
    data_dir : Path
        Directory to check.
    required_gb : float
        Minimum required space in GB.
    """
    try:
        if hasattr(os, "statvfs"):
            stat = os.statvfs(data_dir)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        else:
            # Windows: use shutil.disk_usage
            import shutil
            usage = shutil.disk_usage(data_dir)
            free_gb = usage.free / (1024 ** 3)

        logger.info(f"  Disk space available: {free_gb:.1f} GB")
        if free_gb < required_gb:
            logger.warning(
                f"  WARNING: Only {free_gb:.1f} GB free, need ~{required_gb:.0f} GB. "
                f"Downloads may fail."
            )
    except OSError as e:
        logger.debug(f"  Could not check disk space: {e}")


# =============================================================================
# FILE DOWNLOAD
# =============================================================================

def download_file(
    url: str,
    dest: Path,
    description: str,
    min_size_mb: float,
    verify_ssl: bool = True,
    max_retries: int = 3,
) -> bool:
    """Download a file with resume support, progress output, and retries.

    Parameters
    ----------
    url : str
        URL to download.
    dest : Path
        Destination file path.
    description : str
        Human-readable description for logging.
    min_size_mb : float
        Minimum expected file size in MB for skip-if-exists check.
    verify_ssl : bool
        Whether to verify SSL certificates.
    max_retries : int
        Number of retry attempts on failure.

    Returns
    -------
    bool
        True if download succeeded or file already exists.
    """
    # Skip if file already exists and meets minimum size
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        if size_mb >= min_size_mb:
            logger.info(f"  SKIP {dest.name} ({size_mb:.1f} MB, already exists)")
            return True
        else:
            logger.warning(
                f"  {dest.name} exists but too small ({size_mb:.1f} MB < "
                f"{min_size_mb:.1f} MB), re-downloading"
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  Downloading {description}...")
            logger.debug(f"  URL: {url}")
            logger.debug(f"  Dest: {dest}")

            response = requests.get(
                url, stream=True, timeout=(60, 600), verify=verify_ssl
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            total_mb = total_size / (1024 * 1024) if total_size else 0
            downloaded = 0
            last_progress = 0
            chunk_size = 1024 * 1024  # 1 MB chunks

            # Check for HTML response (e.g., Illumina redirect to terms page)
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type and not url.endswith(".csv"):
                logger.error(
                    f"  ERROR: Got HTML response instead of data file. "
                    f"URL may require manual acceptance of terms: {url}"
                )
                response.close()
                return False

            with open(partial, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress every 10% or 100 MB
                        if total_size > 0:
                            pct = int(downloaded / total_size * 100)
                            if pct >= last_progress + 10:
                                logger.info(
                                    f"    {pct}% ({downloaded / (1024**2):.0f} / "
                                    f"{total_mb:.0f} MB)"
                                )
                                last_progress = pct
                        else:
                            dl_mb = downloaded / (1024 * 1024)
                            if dl_mb >= last_progress + 100:
                                logger.info(f"    {dl_mb:.0f} MB downloaded...")
                                last_progress = dl_mb

            # Rename partial to final
            if partial.exists():
                if dest.exists():
                    dest.unlink()
                partial.rename(dest)

            final_mb = dest.stat().st_size / (1024 * 1024)
            logger.info(f"  OK {dest.name} ({final_mb:.1f} MB)")
            return True

        except KeyboardInterrupt:
            logger.warning(f"\n  Interrupted. Cleaning up partial file...")
            if partial.exists():
                partial.unlink()
            raise

        except Exception as e:
            logger.warning(f"  Attempt {attempt}/{max_retries} failed: {e}")
            if partial.exists():
                partial.unlink()

            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  FAILED to download {description} after {max_retries} attempts")
                return False

    return False


# =============================================================================
# ZIP EXTRACTION
# =============================================================================

def extract_zip(zip_path: Path, member_name: str, extract_to: Path) -> bool:
    """Extract a single member from a zip archive.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file.
    member_name : str
        Name of the member to extract.
    extract_to : Path
        Destination path for the extracted file.

    Returns
    -------
    bool
        True if extraction succeeded.
    """
    if extract_to.exists():
        size_mb = extract_to.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            logger.info(f"  SKIP extract {extract_to.name} ({size_mb:.1f} MB, already exists)")
            return True

    try:
        logger.info(f"  Extracting {member_name} from {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the member (may be in a subdirectory)
            members = zf.namelist()
            target = None
            for m in members:
                if m.endswith(member_name) or m == member_name:
                    target = m
                    break

            if target is None:
                logger.error(
                    f"  ERROR: {member_name} not found in {zip_path.name}. "
                    f"Contents: {members}"
                )
                return False

            extract_to.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(target) as src, open(extract_to, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

        size_mb = extract_to.stat().st_size / (1024 * 1024)
        logger.info(f"  OK extracted {extract_to.name} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"  ERROR extracting {member_name}: {e}")
        return False


# =============================================================================
# VALIDATION
# =============================================================================

def validate_gzip_header(path: Path) -> bool:
    """Validate a gzip file by reading its first line.

    Parameters
    ----------
    path : Path
        Path to gzip file.

    Returns
    -------
    bool
        True if file is a valid gzip with readable content.
    """
    try:
        with gzip.open(path, "rt", errors="replace") as f:
            first_line = f.readline()
        if len(first_line) > 0:
            logger.debug(f"  Validated gzip: {path.name} (first line: {len(first_line)} chars)")
            return True
        logger.warning(f"  Validation: {path.name} gzip is empty")
        return False
    except Exception as e:
        logger.warning(f"  Validation failed for {path.name}: {e}")
        return False


def validate_csv_header(path: Path, expected_substring: str) -> bool:
    """Check that a CSV file header contains an expected substring.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    expected_substring : str
        Substring expected in the header row.

    Returns
    -------
    bool
        True if the header contains the expected substring.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
        if expected_substring.lower() in header.lower():
            logger.debug(f"  Validated CSV header: {path.name}")
            return True
        logger.warning(
            f"  Validation: {path.name} header does not contain '{expected_substring}'"
        )
        return False
    except Exception as e:
        logger.warning(f"  Validation failed for {path.name}: {e}")
        return False


def validate_csv_header_multiline(
    path: Path, expected_substring: str, max_lines: int = 10
) -> bool:
    """Check that one of the first N lines of a CSV contains an expected substring.

    Useful for files like the Illumina manifest that have metadata rows
    before the actual column header.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    expected_substring : str
        Substring expected in a header row.
    max_lines : int
        Number of lines to search.

    Returns
    -------
    bool
        True if any of the first N lines contains the expected substring.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i in range(max_lines):
                line = f.readline()
                if not line:
                    break
                if expected_substring.lower() in line.lower():
                    logger.debug(
                        f"  Validated CSV header: {path.name} "
                        f"(found '{expected_substring}' on line {i + 1})"
                    )
                    return True
        logger.warning(
            f"  Validation: {path.name} first {max_lines} lines do not "
            f"contain '{expected_substring}'"
        )
        return False
    except Exception as e:
        logger.warning(f"  Validation failed for {path.name}: {e}")
        return False


def validate_tsv_header(path: Path) -> bool:
    """Check that a TSV file has a readable header with tabs.

    Parameters
    ----------
    path : Path
        Path to TSV file.

    Returns
    -------
    bool
        True if file has a readable tab-delimited header.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
        if "\t" in header and len(header) > 10:
            logger.debug(f"  Validated TSV header: {path.name}")
            return True
        logger.warning(f"  Validation: {path.name} does not look like a TSV file")
        return False
    except Exception as e:
        logger.warning(f"  Validation failed for {path.name}: {e}")
        return False


def validate_file(key: str, path: Path) -> bool:
    """Dispatch validation to the appropriate checker per file type.

    Parameters
    ----------
    key : str
        Download key from DOWNLOADS dict.
    path : Path
        Path to the downloaded file.

    Returns
    -------
    bool
        True if validation passed.
    """
    if not path.exists():
        logger.warning(f"  Validation: {path.name} does not exist")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)

    if key == "methylation":
        return size_mb >= 3500 and validate_gzip_header(path)
    elif key == "expression":
        # Validate extracted .txt, not the .zip
        return size_mb >= 200 and validate_tsv_header(path)
    elif key == "drug_response":
        return size_mb >= 20
    elif key == "compounds":
        return validate_csv_header(path, "DRUG_NAME")
    elif key == "cell_lines":
        return size_mb >= 0.01
    elif key == "string_ppi":
        return size_mb >= 50 and validate_gzip_header(path)
    elif key == "string_info":
        return size_mb >= 1 and validate_gzip_header(path)
    elif key == "illumina_manifest":
        # Illumina manifest has metadata rows before the actual header,
        # so we search the first 10 lines for IlmnID instead of just line 1
        return size_mb >= 100 and validate_csv_header_multiline(path, "IlmnID", max_lines=10)

    return True


# =============================================================================
# PUBCHEM SMILES
# =============================================================================

def clean_drug_name(name: str) -> str:
    """Clean a drug name for PubChem lookup.

    Strips salt forms, parenthetical aliases, and normalizes whitespace.

    Parameters
    ----------
    name : str
        Raw drug name from GDSC.

    Returns
    -------
    str
        Cleaned drug name.
    """
    cleaned = name.strip()
    # Remove parenthetical aliases like "(compound X)"
    import re
    cleaned = re.sub(r"\s*\(.*?\)\s*", " ", cleaned).strip()
    # Remove common salt suffixes
    for suffix in [" hydrochloride", " HCl", " dihydrochloride", " mesylate",
                   " maleate", " tosylate", " sodium", " potassium",
                   " fumarate", " succinate", " citrate", " tartrate",
                   " acetate", " sulfate", " phosphate", " bromide",
                   " chloride", " nitrate"]:
        if cleaned.lower().endswith(suffix.lower()):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned


def get_drug_list_from_compounds(compounds_path: Path) -> list:
    """Parse GDSC compounds CSV to extract drug names and PubChem CIDs.

    Parameters
    ----------
    compounds_path : Path
        Path to GDSC screened_compounds CSV.

    Returns
    -------
    list of dict
        Each dict has 'name' and optionally 'pubchem_cid'.
    """
    drugs = []
    try:
        with open(compounds_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            logger.debug(f"  Compounds CSV headers: {headers}")

            # Detect column names (may vary between releases)
            name_col = None
            cid_col = None
            for h in headers:
                h_lower = h.lower().strip()
                if "drug_name" in h_lower or h_lower == "name":
                    name_col = h
                if "pubchem" in h_lower or "cid" in h_lower:
                    cid_col = h

            if name_col is None:
                # Fall back to first column
                name_col = headers[0]
                logger.warning(f"  Could not find drug name column, using '{name_col}'")

            seen = set()
            for row in reader:
                name = row.get(name_col, "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)

                cid = None
                if cid_col and row.get(cid_col, "").strip():
                    try:
                        cid = int(float(row[cid_col].strip()))
                    except (ValueError, TypeError):
                        pass

                drugs.append({"name": name, "pubchem_cid": cid})

        logger.info(f"  Found {len(drugs)} unique drugs in compounds file")
        return drugs

    except Exception as e:
        logger.error(f"  ERROR reading compounds file: {e}")
        return []


def fetch_smiles_for_drug(
    name: str, cid: int, session: requests.Session
) -> str:
    """Fetch SMILES string for a drug from PubChem.

    Tries by CID first, then exact name, then cleaned name.

    Parameters
    ----------
    name : str
        Drug name.
    cid : int or None
        PubChem CID if known.
    session : requests.Session
        Requests session for connection pooling.

    Returns
    -------
    str or None
        Canonical SMILES string, or None if not found.
    """
    def _try_url(url, label):
        try:
            resp = session.get(url, timeout=PUBCHEM_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
                logger.debug(f"    {name}: found by {label}")
                return smiles
            elif resp.status_code == 404:
                logger.debug(f"    {name}: not found by {label} (404)")
            else:
                logger.debug(
                    f"    {name}: {label} returned HTTP {resp.status_code}"
                )
        except requests.exceptions.ConnectionError as e:
            logger.debug(f"    {name}: connection error ({label}): {e}")
        except requests.exceptions.Timeout:
            logger.debug(f"    {name}: timeout ({label})")
        except Exception as e:
            logger.debug(f"    {name}: error ({label}): {type(e).__name__}: {e}")
        return None

    # Strategy 1: By CID
    if cid:
        url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        result = _try_url(url, f"CID {cid}")
        if result:
            return result

    # Strategy 2: Exact name
    url = f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(name)}/property/CanonicalSMILES/JSON"
    result = _try_url(url, "exact name")
    if result:
        return result

    # Strategy 3: Cleaned name
    cleaned = clean_drug_name(name)
    if cleaned != name:
        url = f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(cleaned)}/property/CanonicalSMILES/JSON"
        result = _try_url(url, f"cleaned name '{cleaned}'")
        if result:
            return result

    return None


def download_drug_smiles(
    compounds_path: Path, output_path: Path, data_dir: Path
) -> bool:
    """Download SMILES for all GDSC drugs via PubChem API.

    Uses a JSON cache file for resume support across runs.

    Parameters
    ----------
    compounds_path : Path
        Path to GDSC screened_compounds CSV.
    output_path : Path
        Path to output drug_smiles.csv.
    data_dir : Path
        Data directory root (for cache file placement).

    Returns
    -------
    bool
        True if SMILES file was created successfully.
    """
    cache_path = data_dir / "drug_smiles.cache.json"

    # Load existing cache
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            logger.info(f"  Loaded SMILES cache: {len(cache)} entries")
        except Exception:
            cache = {}

    # Get drug list
    drugs = get_drug_list_from_compounds(compounds_path)
    if not drugs:
        logger.error("  No drugs found in compounds file")
        return False

    # Connectivity check: try a known drug before querying all 500+
    logger.info("  Testing PubChem API connectivity...")
    try:
        test_url = f"{PUBCHEM_BASE_URL}/compound/name/aspirin/property/CanonicalSMILES/JSON"
        test_resp = requests.get(test_url, timeout=PUBCHEM_TIMEOUT)
        if test_resp.status_code == 200:
            test_data = test_resp.json()
            test_smiles = test_data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
            logger.info(f"  PubChem API reachable (aspirin -> {test_smiles})")
        else:
            logger.error(
                f"  PubChem API returned HTTP {test_resp.status_code} for test query. "
                f"Response: {test_resp.text[:200]}"
            )
            logger.error("  Skipping SMILES download. Try --skip-smiles or check network.")
            return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"  Cannot reach PubChem API (connection error): {e}")
        logger.error("  Skipping SMILES download. PubChem may be blocked from this network.")
        return False
    except Exception as e:
        logger.error(f"  PubChem API test failed: {type(e).__name__}: {e}")
        logger.error("  Skipping SMILES download.")
        return False

    # Filter to unresolved drugs (not in cache or explicitly None meaning "tried and failed")
    unresolved = [d for d in drugs if d["name"] not in cache]
    logger.info(
        f"  Drugs: {len(drugs)} total, {len(cache)} cached, "
        f"{len(unresolved)} to query"
    )

    if unresolved:
        session = requests.Session()
        session.headers.update({"Accept": "application/json"})

        try:
            consecutive_failures = 0
            for i, drug in enumerate(unresolved):
                name = drug["name"]
                cid = drug.get("pubchem_cid")

                smiles = fetch_smiles_for_drug(name, cid, session)
                cache[name] = smiles  # None if not found

                if smiles:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                # Log first 3 individual failures at INFO level for diagnostics
                if smiles is None and i < 3:
                    logger.info(f"    [{name}] not found (check log for details)")

                # Abort early if first 20 all fail (likely network issue)
                if i == 19 and consecutive_failures == 20:
                    logger.error(
                        "  First 20 drugs all failed. Likely a network/firewall issue. "
                        "Aborting SMILES download. Check the log file for DEBUG details."
                    )
                    break

                if (i + 1) % 10 == 0:
                    found = sum(1 for v in cache.values() if v is not None)
                    logger.info(
                        f"    Progress: {i + 1}/{len(unresolved)} queried "
                        f"({found}/{len(cache)} total found)"
                    )

                # Save cache periodically
                if (i + 1) % PUBCHEM_CACHE_SAVE_INTERVAL == 0:
                    with open(cache_path, "w") as f:
                        json.dump(cache, f, indent=2)

                time.sleep(PUBCHEM_RATE_LIMIT_DELAY)

        except KeyboardInterrupt:
            logger.warning("\n  Interrupted. Saving SMILES cache...")
        finally:
            # Always save cache on exit
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)
            session.close()

    # Write output CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["drug_name", "smiles"])
        for drug in drugs:
            name = drug["name"]
            smiles = cache.get(name)
            if smiles:
                writer.writerow([name, smiles])
                found_count += 1

    logger.info(
        f"  OK drug_smiles.csv: {found_count}/{len(drugs)} drugs with SMILES "
        f"({found_count / len(drugs) * 100:.0f}%)"
    )
    return found_count >= 300


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(results: dict, data_dir: Path) -> None:
    """Print a final summary table of all download results.

    Parameters
    ----------
    results : dict
        Mapping of source key to (status, size_mb) tuples.
    data_dir : Path
        Data directory root.
    """
    logger.info("")
    logger.info("=" * 72)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 72)
    logger.info(f"  {'Source':<25s} {'Status':<10s} {'Size':>10s}")
    logger.info(f"  {'-'*25} {'-'*10} {'-'*10}")

    for key, (status, size_mb) in results.items():
        size_str = f"{size_mb:.1f} MB" if size_mb > 0 else "--"
        logger.info(f"  {key:<25s} {status:<10s} {size_str:>10s}")

    n_ok = sum(1 for s, _ in results.values() if s == "OK")
    n_total = len(results)
    logger.info(f"  {'-'*25} {'-'*10} {'-'*10}")
    logger.info(f"  {'Total':<25s} {n_ok}/{n_total}")
    logger.info("=" * 72)

    if n_ok < n_total:
        logger.info(
            f"\n  Some downloads failed. Check log: {data_dir / 'download_all_data.log'}"
        )
    else:
        logger.info("\n  All downloads completed successfully.")

    # Note about path change for integrate_multiomics.py
    expr_dest = data_dir / "raw" / "GDSC" / "Cell_line_RMA_proc_basalExp.txt"
    if expr_dest.exists():
        logger.info(
            f"\n  NOTE: Gene expression is at {expr_dest} "
            f"(integrate_multiomics.py referenced data/raw/CCLE/; update path if needed)"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Orchestrate all downloads."""
    args = parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(data_dir / "download_all_data.log")

    logger.info("=" * 72)
    logger.info("Cancer Drug Response Prediction - Data Download")
    logger.info("=" * 72)
    logger.info(f"  Data directory: {data_dir}")

    verify_ssl = not args.no_verify_ssl
    if not verify_ssl:
        logger.warning("  SSL verification disabled")
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    check_disk_space(data_dir)

    # Determine which sources to download
    if args.only:
        selected_keys = [k for k in args.only if k in DOWNLOADS]
        do_smiles = "smiles" in args.only and not args.skip_smiles
    else:
        selected_keys = list(DOWNLOADS.keys())
        do_smiles = not args.skip_smiles

    results = {}

    # Download each source
    for key in selected_keys:
        info = DOWNLOADS[key]
        dest = data_dir / info["dest"]

        logger.info("")
        logger.info(f"[{key}] {info['description']}")

        ok = download_file(
            url=info["url"],
            dest=dest,
            description=info["description"],
            min_size_mb=info["min_size_mb"],
            verify_ssl=verify_ssl,
        )

        # Handle zip extraction
        if ok and "extract" in info:
            extract_info = info["extract"]
            extract_dest = data_dir / extract_info["dest"]
            ok = extract_zip(dest, extract_info["member"], extract_dest)

        # Validation
        if ok and not args.no_validate:
            if "extract" in info:
                validate_path = data_dir / info["extract"]["dest"]
                # Use the "expression" key for extracted file validation
            else:
                validate_path = dest

            valid = validate_file(key, validate_path)
            if not valid:
                logger.warning(f"  WARNING: {key} validation failed (file may still be usable)")

        size_mb = 0
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
        results[key] = ("OK" if ok else "FAILED", size_mb)

    # PubChem SMILES
    if do_smiles:
        logger.info("")
        logger.info("[smiles] Drug SMILES from PubChem API")

        compounds_path = data_dir / DOWNLOADS["compounds"]["dest"]
        smiles_output = data_dir / "processed" / "drug_smiles.csv"

        if not compounds_path.exists():
            logger.error(
                "  Cannot download SMILES: compounds file not found. "
                "Download compounds first."
            )
            results["smiles"] = ("FAILED", 0)
        else:
            ok = download_drug_smiles(compounds_path, smiles_output, data_dir)
            size_mb = 0
            if smiles_output.exists():
                size_mb = smiles_output.stat().st_size / (1024 * 1024)
            results["smiles"] = ("OK" if ok else "PARTIAL", size_mb)

    print_summary(results, data_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
