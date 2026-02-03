"""
Data loading utilities for GEO datasets.

This module handles downloading and initial processing of GEO datasets
for the pharmacoepigenetics study.
"""

import GEOparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)

# Define data directories
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_geo_dataset(
    geo_id: str,
    destdir: Optional[Path] = None,
    silent: bool = False
) -> GEOparse.GEOTypes.GSE:
    """
    Download a GEO dataset using GEOparse.

    Parameters
    ----------
    geo_id : str
        GEO accession number (e.g., 'GSE270494')
    destdir : Path, optional
        Destination directory for downloaded files.
        Defaults to RAW_DATA_DIR / geo_id
    silent : bool, default=False
        If True, suppress download progress messages

    Returns
    -------
    GEOparse.GEOTypes.GSE
        The downloaded GEO dataset object

    Notes
    -----
    This function downloads the complete GEO dataset including:
    - Series matrix files
    - Sample metadata
    - Platform annotations

    Examples
    --------
    >>> gse = download_geo_dataset('GSE270494')
    >>> print(f"Downloaded {len(gse.gsms)} samples")
    """
    if destdir is None:
        destdir = RAW_DATA_DIR / geo_id

    destdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {geo_id} to {destdir}")

    try:
        gse = GEOparse.get_GEO(
            geo=geo_id,
            destdir=str(destdir),
            silent=silent
        )
        logger.info(f"Successfully downloaded {geo_id}")
        logger.info(f"Number of samples: {len(gse.gsms)}")
        logger.info(f"Platforms: {list(gse.gpls.keys())}")

        return gse

    except Exception as e:
        logger.error(f"Error downloading {geo_id}: {str(e)}")
        raise


def download_supplementary_files(
    gse: GEOparse.GEOTypes.GSE,
    destdir: Path
) -> None:
    """
    Download supplementary files from GEO dataset.

    Parameters
    ----------
    gse : GEOparse.GEOTypes.GSE
        GEO dataset object
    destdir : Path
        Destination directory for supplementary files
    """
    logger.info("Downloading supplementary files using GEOparse...")

    try:
        # Use GEOparse's built-in download method
        gse.download_supplementary_files(
            directory=str(destdir),
            download_sra=False
        )
        logger.info("  Successfully downloaded supplementary files")
    except Exception as e:
        logger.warning(f"  GEOparse download failed: {str(e)}")
        logger.info("  Attempting manual download...")

        import requests
        from tqdm import tqdm

        suppl_files = gse.metadata.get('supplementary_file', [])

        for file_url in suppl_files:
            # Convert FTP to HTTP
            if file_url.startswith('ftp://'):
                file_url = file_url.replace('ftp://', 'https://')

            filename = file_url.split('/')[-1]
            filepath = destdir / filename

            if filepath.exists():
                logger.info(f"  {filename} already exists, skipping")
                continue

            logger.info(f"  Downloading {filename}...")
            try:
                response = requests.get(file_url, stream=True, timeout=60)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                logger.info(f"    Successfully downloaded {filename}")
            except Exception as download_error:
                logger.error(f"    Error downloading {filename}: {str(download_error)}")


def extract_methylation_matrix(
    geo_id: str,
    species: str = 'human',
    filter_detection_pvals: bool = True
) -> pd.DataFrame:
    """
    Extract DNA methylation beta-values from supplementary files.

    Parameters
    ----------
    geo_id : str
        GEO accession number
    species : str, default='human'
        Species to extract ('human' or 'mouse')
    filter_detection_pvals : bool, default=True
        If True, filter out detection p-value columns to return only beta-values

    Returns
    -------
    pd.DataFrame
        DataFrame with CpG sites as rows and samples as columns
        Values are methylation beta-values (0-1)

    Notes
    -----
    Beta-values represent the ratio of methylated probe intensity to
    total probe intensity (methylated + unmethylated).

    For GSE270494, the raw file contains alternating columns:
    - Sample name (beta-values)
    - Sample name Detection Pval (detection p-values)

    By default, only beta-value columns are returned.
    """
    logger.info(f"Extracting {species} methylation matrix from supplementary files")

    data_dir = RAW_DATA_DIR / geo_id

    # Determine which file to load based on species
    if species.lower() == 'human':
        filename = f"{geo_id}_Noguera-Castells_Average_Beta_Homo_Sapiens.csv.gz"
    elif species.lower() == 'mouse':
        filename = f"{geo_id}_Noguera-Castells_Average_Beta_Mus_Musculus.csv.gz"
    else:
        raise ValueError(f"Unknown species: {species}. Must be 'human' or 'mouse'")

    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Methylation file not found: {filepath}\n"
            f"Please ensure supplementary files have been downloaded."
        )

    # Load the methylation matrix
    logger.info(f"Loading {filename}...")
    df_methylation = pd.read_csv(filepath, index_col=0)

    logger.info(f"Raw data shape: {df_methylation.shape}")
    logger.info(f"  CpG sites: {df_methylation.shape[0]:,}")
    logger.info(f"  Total columns: {df_methylation.shape[1]:,}")

    # Filter out detection p-value columns if requested
    if filter_detection_pvals:
        # Keep only columns that don't contain "Detection Pval"
        beta_cols = [col for col in df_methylation.columns if 'Detection Pval' not in col]
        df_methylation = df_methylation[beta_cols]

        logger.info(f"Filtered to beta-values only: {df_methylation.shape}")
        logger.info(f"  CpG sites: {df_methylation.shape[0]:,}")
        logger.info(f"  Samples: {df_methylation.shape[1]:,}")

    # Validate beta-values are in [0,1] range
    min_val = df_methylation.min().min()
    max_val = df_methylation.max().max()
    mean_val = df_methylation.mean().mean()

    if min_val < 0 or max_val > 1:
        logger.warning(f"Beta-values outside [0,1] range: [{min_val:.4f}, {max_val:.4f}]")
    else:
        logger.info(f"Beta-value validation passed: [{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}")

    return df_methylation


def extract_sample_metadata(
    gse: GEOparse.GEOTypes.GSE
) -> pd.DataFrame:
    """
    Extract sample metadata/phenotype data from GEO dataset.

    Parameters
    ----------
    gse : GEOparse.GEOTypes.GSE
        GEO dataset object

    Returns
    -------
    pd.DataFrame
        DataFrame with samples as rows and metadata columns

    Notes
    -----
    Metadata typically includes:
    - Sample identifiers
    - Disease type/cancer type
    - Cell line name
    - Platform information
    - Processing details
    """
    logger.info("Extracting sample metadata")

    metadata_dict = {}

    for gsm_name, gsm in gse.gsms.items():
        metadata_dict[gsm_name] = {
            'title': gsm.metadata.get('title', [''])[0],
            'source_name': gsm.metadata.get('source_name_ch1', [''])[0],
            'organism': gsm.metadata.get('organism_ch1', [''])[0],
            'characteristics': gsm.metadata.get('characteristics_ch1', []),
        }

    df_metadata = pd.DataFrame.from_dict(metadata_dict, orient='index')

    logger.info(f"Extracted metadata for {len(df_metadata)} samples")

    return df_metadata


def load_gse270494(species: str = 'human') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GSE270494 - Hematological malignancy DNA methylation database.

    Parameters
    ----------
    species : str, default='human'
        Which species data to load ('human' or 'mouse')

    Returns
    -------
    df_methylation : pd.DataFrame
        DNA methylation beta-values matrix
    df_metadata : pd.DataFrame
        Sample metadata

    Notes
    -----
    Dataset details:
    - 210 hematological malignancy cell lines (180 human, 30 mouse)
    - Platform: Infinium HumanMethylation450 BeadChip
    - ~850,000 CpG sites (human), ~285,000 (mouse)
    """
    logger.info("=" * 80)
    logger.info(f"Loading GSE270494: Hematological Malignancy Dataset ({species})")
    logger.info("=" * 80)

    geo_id = 'GSE270494'
    gse = download_geo_dataset(geo_id)

    # Download supplementary files if needed
    data_dir = RAW_DATA_DIR / geo_id
    download_supplementary_files(gse, data_dir)

    # Extract methylation matrix
    df_methylation = extract_methylation_matrix(geo_id, species=species)

    # Extract metadata
    df_metadata = extract_sample_metadata(gse)

    return df_methylation, df_metadata


def load_gse68379() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GSE68379 - Cancer cell line pharmacoepigenomics dataset.

    Returns
    -------
    df_methylation : pd.DataFrame
        DNA methylation beta-values matrix
    df_metadata : pd.DataFrame
        Sample metadata

    Notes
    -----
    Dataset details:
    - 721 cancer cell lines across 22 cancer types
    - Platform: Infinium HumanMethylation450 BeadChip
    - Associated with drug response data from GDSC
    """
    logger.info("=" * 80)
    logger.info("Loading GSE68379: Cancer Cell Line Pharmacoepigenomics")
    logger.info("=" * 80)

    gse = download_geo_dataset('GSE68379')
    df_methylation = extract_methylation_matrix(gse)
    df_metadata = extract_sample_metadata(gse)

    return df_methylation, df_metadata


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading functions...")

    # Load GSE270494
    df_meth_270494, df_meta_270494 = load_gse270494()
    print(f"\nGSE270494 - Methylation shape: {df_meth_270494.shape}")
    print(f"GSE270494 - Metadata shape: {df_meta_270494.shape}")

    # Load GSE68379
    df_meth_68379, df_meta_68379 = load_gse68379()
    print(f"\nGSE68379 - Methylation shape: {df_meth_68379.shape}")
    print(f"GSE68379 - Metadata shape: {df_meta_68379.shape}")
