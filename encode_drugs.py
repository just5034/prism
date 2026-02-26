"""
Encode Drugs: ECFP-4 Fingerprints + Publication-Quality Figures
================================================================

Generates molecular fingerprint encodings from SMILES strings and creates
visualizations of the drug landscape for the pharmacoepigenomics pipeline.

Outputs:
  - ECFP-4 fingerprints (2048-bit Morgan fingerprints, radius=2)
  - One-hot drug encoding (375-dim baseline)
  - Drug index mapping with metadata
  - 9 publication-quality figures

Usage:
    python encode_drugs.py                          # run everything
    python encode_drugs.py --data-dir data          # custom data root
    python encode_drugs.py --no-figures             # encode only, skip plots

Prerequisites:
    pip install rdkit numpy pandas matplotlib seaborn scikit-learn
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("encode_drugs")

# =============================================================================
# CONSTANTS
# =============================================================================

ECFP_RADIUS = 2       # Morgan fingerprint radius (ECFP-4 = radius 2)
ECFP_NBITS = 2048     # Fingerprint bit length
RANDOM_STATE = 42      # Reproducibility seed
TSNE_PERPLEXITY = 30   # t-SNE perplexity
MIN_PATHWAY_SIZE = 3   # Pathways with fewer drugs grouped into "Other"
FIGURE_DPI = 300       # Publication-quality DPI


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

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

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
        description="Generate drug encodings (ECFP-4 + one-hot) and figures.",
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
        "--no-figures",
        action="store_true",
        help="Skip figure generation (encode only)",
    )
    return parser.parse_args()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_smiles(data_dir: Path) -> pd.DataFrame:
    """Load drug SMILES from CSV.

    Parameters
    ----------
    data_dir : Path
        Root data directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: drug_name, smiles.
    """
    path = data_dir / "processed" / "drug_smiles.csv"
    df_smiles = pd.read_csv(path, dtype=str)
    df_smiles = df_smiles.dropna(subset=["smiles"])
    df_smiles = df_smiles.drop_duplicates(subset=["drug_name"])
    logger.info(f"  Loaded {len(df_smiles)} drugs with SMILES from {path.name}")
    return df_smiles


def load_compounds(data_dir: Path) -> pd.DataFrame:
    """Load GDSC compound annotations.

    Parameters
    ----------
    data_dir : Path
        Root data directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with GDSC compound metadata.
    """
    path = data_dir / "raw" / "GDSC" / "screened_compounds_rel_8.5.csv"
    df_compounds = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    logger.info(f"  Loaded {len(df_compounds)} compounds from {path.name}")
    logger.debug(f"  Compound columns: {list(df_compounds.columns)}")
    return df_compounds


def load_available_drugs(data_dir: Path) -> pd.DataFrame:
    """Load the 375 modeled drugs list.

    Parameters
    ----------
    data_dir : Path
        Root data directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Rank, Drug, Samples, Percent_Complete.
    """
    path = data_dir / "processed" / "available_drugs.csv"
    df_avail = pd.read_csv(path)
    logger.info(f"  Loaded {len(df_avail)} modeled drugs from {path.name}")
    return df_avail


def load_sample_metadata(data_dir: Path) -> pd.DataFrame | None:
    """Load GSE68379 sample metadata if available.

    Parameters
    ----------
    data_dir : Path
        Root data directory.

    Returns
    -------
    pd.DataFrame or None
        Sample metadata, or None if file not found.
    """
    path = data_dir / "processed" / "GSE68379_sample_metadata.csv"
    if not path.exists():
        logger.warning(f"  Sample metadata not found at {path}, skipping panel (a)")
        return None
    df_meta = pd.read_csv(path)
    logger.info(f"  Loaded {len(df_meta)} samples from {path.name}")
    return df_meta


# =============================================================================
# DRUG MATCHING & COVERAGE
# =============================================================================

def match_drugs(
    df_avail: pd.DataFrame,
    df_smiles: pd.DataFrame,
    df_compounds: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-reference modeled drugs with SMILES and compound metadata.

    Parameters
    ----------
    df_avail : pd.DataFrame
        Modeled drugs (375).
    df_smiles : pd.DataFrame
        Drugs with SMILES.
    df_compounds : pd.DataFrame
        GDSC compound annotations.

    Returns
    -------
    pd.DataFrame
        Drug index with columns: drug_name, has_smiles, target, pathway,
        rank, samples.
    """
    # Build lookup dicts
    smiles_set = set(df_smiles["drug_name"].str.strip())

    compound_meta = {}
    for _, row in df_compounds.iterrows():
        name = str(row.get("DRUG_NAME", "")).strip()
        if name:
            compound_meta[name] = {
                "target": str(row.get("TARGET", "")).strip(),
                "pathway": str(row.get("TARGET_PATHWAY", "")).strip(),
            }

    records = []
    for _, row in df_avail.iterrows():
        drug = str(row["Drug"]).strip()
        meta = compound_meta.get(drug, {"target": "", "pathway": ""})
        records.append({
            "drug_name": drug,
            "has_smiles": drug in smiles_set,
            "target": meta["target"],
            "pathway": meta["pathway"],
            "rank": int(row["Rank"]),
            "samples": int(row["Samples"]),
        })

    df_index = pd.DataFrame(records)

    n_with = df_index["has_smiles"].sum()
    n_total = len(df_index)
    logger.info(f"  {n_with}/{n_total} modeled drugs have SMILES ({n_with/n_total*100:.0f}%)")

    missing = df_index[~df_index["has_smiles"]]
    if len(missing) > 0:
        logger.info(f"  Missing SMILES ({len(missing)} drugs):")
        for _, r in missing.iterrows():
            tgt = r["target"][:40] if r["target"] else "(no target)"
            logger.info(f"    {r['drug_name']:<35s} {tgt}")

    return df_index


# =============================================================================
# FINGERPRINT GENERATION
# =============================================================================

def generate_ecfp(
    df_smiles: pd.DataFrame,
    df_index: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Generate ECFP-4 fingerprints for drugs with valid SMILES.

    Parameters
    ----------
    df_smiles : pd.DataFrame
        Drug names and SMILES strings.
    df_index : pd.DataFrame
        Drug index (to filter to modeled drugs).

    Returns
    -------
    tuple of (np.ndarray, list of str)
        Fingerprint array shape (N, 2048) and corresponding drug names.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Build SMILES lookup
    smiles_map = dict(zip(df_smiles["drug_name"].str.strip(), df_smiles["smiles"]))

    # Filter to modeled drugs with SMILES
    drugs_with_smiles = df_index[df_index["has_smiles"]]["drug_name"].tolist()

    fingerprints = []
    valid_names = []
    failed = []

    for drug_name in drugs_with_smiles:
        smi = smiles_map.get(drug_name)
        if not smi:
            failed.append((drug_name, "no SMILES in lookup"))
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed.append((drug_name, f"RDKit parse failure: {smi[:60]}"))
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=ECFP_RADIUS, nBits=ECFP_NBITS
        )
        arr = np.zeros(ECFP_NBITS, dtype=np.float32)
        for bit in fp.GetOnBits():
            arr[bit] = 1.0

        fingerprints.append(arr)
        valid_names.append(drug_name)

    if failed:
        logger.warning(f"  {len(failed)} drugs failed SMILES parsing:")
        for name, reason in failed:
            logger.warning(f"    {name}: {reason}")

    arr_ecfp = np.stack(fingerprints) if fingerprints else np.empty((0, ECFP_NBITS))
    logger.info(f"  Generated ECFP-4 fingerprints: {arr_ecfp.shape}")

    # Log bit density stats
    if arr_ecfp.shape[0] > 0:
        bits_on = arr_ecfp.sum(axis=1)
        logger.info(
            f"  Bits on per drug: mean={bits_on.mean():.0f}, "
            f"min={bits_on.min():.0f}, max={bits_on.max():.0f}"
        )

    return arr_ecfp, valid_names


def generate_onehot(n_drugs: int) -> np.ndarray:
    """Generate one-hot encoding for all modeled drugs.

    Parameters
    ----------
    n_drugs : int
        Number of modeled drugs (375).

    Returns
    -------
    np.ndarray
        Identity matrix of shape (n_drugs, n_drugs), float32.
    """
    arr_onehot = np.eye(n_drugs, dtype=np.float32)
    logger.info(f"  Generated one-hot encoding: {arr_onehot.shape}")
    return arr_onehot


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_encodings(
    arr_ecfp: np.ndarray,
    arr_onehot: np.ndarray,
    ecfp_names: list[str],
    df_index: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save encodings and index to disk.

    Parameters
    ----------
    arr_ecfp : np.ndarray
        ECFP fingerprint array (N, 2048).
    arr_onehot : np.ndarray
        One-hot encoding array (375, 375).
    ecfp_names : list of str
        Drug names corresponding to ECFP rows.
    df_index : pd.DataFrame
        Full drug index with metadata.
    output_dir : Path
        Directory for embedding outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ECFP fingerprints
    ecfp_path = output_dir / "drug_ecfp.npy"
    np.save(ecfp_path, arr_ecfp)
    logger.info(f"  Saved {ecfp_path} {arr_ecfp.shape}")

    # Save one-hot
    onehot_path = output_dir / "drug_onehot.npy"
    np.save(onehot_path, arr_onehot)
    logger.info(f"  Saved {onehot_path} {arr_onehot.shape}")

    # Build drug index CSV with row mappings
    ecfp_name_to_idx = {name: i for i, name in enumerate(ecfp_names)}
    df_out = df_index.copy()
    df_out["ecfp_row_idx"] = df_out["drug_name"].map(ecfp_name_to_idx)
    df_out["onehot_idx"] = df_out["rank"] - 1  # rank is 1-based

    # Reorder columns
    cols = ["drug_name", "ecfp_row_idx", "onehot_idx", "has_smiles",
            "target", "pathway", "rank", "samples"]
    df_out = df_out[[c for c in cols if c in df_out.columns]]

    index_path = output_dir / "drug_index.csv"
    df_out.to_csv(index_path, index=False)
    logger.info(f"  Saved {index_path} ({len(df_out)} drugs)")


# =============================================================================
# TANIMOTO SIMILARITY
# =============================================================================

def compute_tanimoto_matrix(arr_ecfp: np.ndarray) -> np.ndarray:
    """Compute pairwise Tanimoto similarity from binary fingerprints.

    Uses vectorized bitwise operations for speed.

    Parameters
    ----------
    arr_ecfp : np.ndarray
        Binary fingerprint array of shape (N, D).

    Returns
    -------
    np.ndarray
        Symmetric similarity matrix of shape (N, N), values in [0, 1].
    """
    # Convert to bool for bitwise ops
    bits = arr_ecfp.astype(bool)

    # Bit counts per molecule
    counts = bits.sum(axis=1)  # (N,)

    # Intersection: A & B counts via dot product of binary matrix
    intersection = bits.astype(np.float32) @ bits.astype(np.float32).T  # (N, N)

    # Union = |A| + |B| - |A & B|
    union = counts[:, None] + counts[None, :] - intersection

    # Tanimoto = intersection / union (handle zero division)
    with np.errstate(divide="ignore", invalid="ignore"):
        tanimoto = np.where(union > 0, intersection / union, 0.0)

    return tanimoto.astype(np.float32)


# =============================================================================
# FIGURE HELPERS
# =============================================================================

def _get_pathway_colors(
    pathways: pd.Series,
) -> tuple[dict[str, str], pd.Series]:
    """Assign colors to pathways, grouping rare ones into 'Other'.

    Parameters
    ----------
    pathways : pd.Series
        Pathway labels for each drug.

    Returns
    -------
    tuple of (dict, pd.Series)
        Color mapping and cleaned pathway series.
    """
    import matplotlib.pyplot as plt

    # Group rare pathways into "Other"
    counts = pathways.value_counts()
    rare = set(counts[counts < MIN_PATHWAY_SIZE].index)
    cleaned = pathways.apply(lambda x: "Other" if x in rare or not x else x)

    # Assign colors
    unique = sorted(cleaned.unique())
    # Put "Other" last
    if "Other" in unique:
        unique.remove("Other")
        unique.append("Other")

    n_colors = len(unique)
    if n_colors <= 20:
        cmap = plt.cm.tab20
        colors = [cmap(i / 20) for i in range(n_colors)]
    else:
        cmap = plt.cm.gist_ncar
        colors = [cmap(i / n_colors) for i in range(n_colors)]

    # Make "Other" gray
    color_map = {}
    for i, pw in enumerate(unique):
        if pw == "Other":
            color_map[pw] = "#999999"
        else:
            color_map[pw] = colors[i]

    return color_map, cleaned


def _set_style() -> None:
    """Set matplotlib style for publication figures."""
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("default")
            logger.debug("  Using default matplotlib style")


# =============================================================================
# FIGURE 1: Drug Chemical Space (t-SNE)
# =============================================================================

def plot_tsne(
    arr_ecfp: np.ndarray,
    ecfp_names: list[str],
    df_index: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Plot t-SNE of drug chemical space colored by target pathway.

    Parameters
    ----------
    arr_ecfp : np.ndarray
        ECFP fingerprints (N, 2048).
    ecfp_names : list of str
        Drug names for each row.
    df_index : pd.DataFrame
        Drug metadata with pathway info.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    _set_style()

    logger.info("  Computing t-SNE embedding...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(TSNE_PERPLEXITY, arr_ecfp.shape[0] - 1),
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(arr_ecfp)

    # Map pathway labels to ECFP drugs
    name_to_pathway = dict(zip(df_index["drug_name"], df_index["pathway"]))
    pathways = pd.Series([name_to_pathway.get(n, "") for n in ecfp_names])
    color_map, pathways_clean = _get_pathway_colors(pathways)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot each pathway separately for legend
    for pw in sorted(color_map.keys()):
        mask = pathways_clean == pw
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color_map[pw]],
            label=f"{pw} ({mask.sum()})",
            s=40, alpha=0.75, edgecolors="white", linewidth=0.3,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("Drug Chemical Space (ECFP-4 t-SNE)", fontsize=14, fontweight="bold")

    # Legend outside plot
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=8, frameon=True, ncol=1,
        title="Target Pathway", title_fontsize=9,
    )

    plt.tight_layout()
    path = fig_dir / "drug_chemical_space_tsne.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 2: Drug Similarity Heatmap
# =============================================================================

def plot_similarity_heatmap(
    arr_ecfp: np.ndarray,
    ecfp_names: list[str],
    df_index: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Plot Tanimoto similarity heatmap grouped by target pathway.

    Parameters
    ----------
    arr_ecfp : np.ndarray
        ECFP fingerprints (N, 2048).
    ecfp_names : list of str
        Drug names for each row.
    df_index : pd.DataFrame
        Drug metadata with pathway info.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    _set_style()

    logger.info("  Computing Tanimoto similarity matrix...")
    sim = compute_tanimoto_matrix(arr_ecfp)

    # Sort by pathway for visible block structure
    name_to_pathway = dict(zip(df_index["drug_name"], df_index["pathway"]))
    pathways = [name_to_pathway.get(n, "Unknown") for n in ecfp_names]

    # Sort indices by pathway
    sorted_idx = sorted(range(len(pathways)), key=lambda i: (pathways[i], ecfp_names[i]))
    sim_sorted = sim[np.ix_(sorted_idx, sorted_idx)]
    pathways_sorted = [pathways[i] for i in sorted_idx]

    # Get pathway colors
    pw_series = pd.Series(pathways_sorted)
    color_map, pw_clean = _get_pathway_colors(pw_series)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(sim_sorted, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Tanimoto Similarity")

    # Add pathway color bar on left
    pw_colors = [color_map[p] for p in pw_clean]
    for i, c in enumerate(pw_colors):
        ax.plot(-2, i, "s", color=c, markersize=2, clip_on=False)

    ax.set_xlabel("Drugs (sorted by pathway)", fontsize=11)
    ax.set_ylabel("Drugs (sorted by pathway)", fontsize=11)
    ax.set_title(
        "Drug Pairwise Tanimoto Similarity (ECFP-4)",
        fontsize=14, fontweight="bold",
    )

    # Remove tick labels (too many drugs)
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend for pathways
    unique_pw = sorted(set(pw_clean))
    legend_handles = [
        Patch(facecolor=color_map[pw], label=pw) for pw in unique_pw
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.15, 1), loc="upper left",
        fontsize=7, frameon=True, ncol=1,
        title="Target Pathway", title_fontsize=8,
    )

    plt.tight_layout()
    path = fig_dir / "drug_similarity_heatmap.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 3: Dataset Overview Dashboard
# =============================================================================

def plot_dashboard(
    df_index: pd.DataFrame,
    df_compounds: pd.DataFrame,
    df_avail: pd.DataFrame,
    df_sample_meta: pd.DataFrame | None,
    fig_dir: Path,
) -> None:
    """Plot 2x2 dataset overview dashboard.

    Panels:
      (a) Cell lines by cancer type (top 15)
      (b) Drugs by target pathway
      (c) SMILES coverage
      (d) Drug testing completeness

    Parameters
    ----------
    df_index : pd.DataFrame
        Drug index with metadata.
    df_compounds : pd.DataFrame
        GDSC compound annotations.
    df_avail : pd.DataFrame
        Modeled drugs with sample counts.
    df_sample_meta : pd.DataFrame or None
        Sample metadata for cancer type counts.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _set_style()

    has_meta = df_sample_meta is not None
    if has_meta:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        ax_a, ax_b = axes[0]
        ax_c, ax_d = axes[1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax_b, ax_c, ax_d = axes
        ax_a = None

    # --- Panel (a): Cell lines by cancer type ---
    if has_meta and ax_a is not None:
        # Try common column names for site/cancer type
        site_col = None
        for candidate in ["primary site", "primary_site", "Site", "site",
                          "cancer_type", "Primary Site"]:
            if candidate in df_sample_meta.columns:
                site_col = candidate
                break

        if site_col:
            site_counts = df_sample_meta[site_col].value_counts().head(15)
            colors_a = sns.color_palette("Set3", n_colors=len(site_counts))
            site_counts.plot.barh(ax=ax_a, color=colors_a)
            ax_a.set_xlabel("Number of Cell Lines", fontsize=10)
            ax_a.set_title("(a) Cell Lines by Cancer Type (Top 15)", fontsize=11,
                           fontweight="bold")
            ax_a.invert_yaxis()
        else:
            ax_a.text(0.5, 0.5, "Cancer type column\nnot found in metadata",
                      ha="center", va="center", transform=ax_a.transAxes, fontsize=10)
            ax_a.set_title("(a) Cell Lines by Cancer Type", fontsize=11,
                           fontweight="bold")

    # --- Panel (b): Drugs by target pathway ---
    pathway_counts = df_compounds["TARGET_PATHWAY"].value_counts().head(15)
    colors_b = sns.color_palette("husl", n_colors=len(pathway_counts))
    pathway_counts.plot.barh(ax=ax_b, color=colors_b)
    ax_b.set_xlabel("Number of Drugs", fontsize=10)
    ax_b.set_title("(b) Drugs by Target Pathway (Top 15)", fontsize=11,
                    fontweight="bold")
    ax_b.invert_yaxis()

    # --- Panel (c): SMILES coverage ---
    n_with = df_index["has_smiles"].sum()
    n_without = len(df_index) - n_with
    wedges, texts, autotexts = ax_c.pie(
        [n_with, n_without],
        labels=[f"With SMILES\n({n_with})", f"Without SMILES\n({n_without})"],
        colors=["#2ecc71", "#e74c3c"],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax_c.set_title(
        f"(c) SMILES Coverage ({len(df_index)} Modeled Drugs)",
        fontsize=11, fontweight="bold",
    )

    # --- Panel (d): Drug testing completeness ---
    ax_d.hist(
        df_avail["Samples"], bins=30, color="#3498db", edgecolor="white", alpha=0.85,
    )
    ax_d.axvline(
        df_avail["Samples"].median(), color="#e74c3c", linestyle="--", linewidth=1.5,
        label=f"Median: {df_avail['Samples'].median():.0f}",
    )
    ax_d.set_xlabel("Number of Cell Lines Tested", fontsize=10)
    ax_d.set_ylabel("Number of Drugs", fontsize=10)
    ax_d.set_title("(d) Drug Testing Completeness", fontsize=11, fontweight="bold")
    ax_d.legend(fontsize=9)

    plt.suptitle(
        "Pharmacoepigenomics Dataset Overview",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = fig_dir / "dataset_overview_dashboard.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 4: Pathway Coverage Bar Chart
# =============================================================================

def plot_pathway_coverage(
    df_index: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Plot horizontal bar chart of SMILES coverage by target pathway.

    Parameters
    ----------
    df_index : pd.DataFrame
        Drug index with has_smiles and pathway columns.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt

    _set_style()

    # Count per pathway
    pathways = df_index[df_index["pathway"] != ""]["pathway"]
    if pathways.empty:
        logger.warning("  No pathway data available, skipping pathway coverage plot")
        return

    pw_total = df_index.groupby("pathway").size()
    pw_with = df_index[df_index["has_smiles"]].groupby("pathway").size()

    # Combine into DataFrame
    df_pw = pd.DataFrame({
        "total": pw_total,
        "with_smiles": pw_with,
    }).fillna(0).astype(int)
    df_pw["without_smiles"] = df_pw["total"] - df_pw["with_smiles"]
    df_pw = df_pw.sort_values("total", ascending=True)

    # Remove empty pathway row if present
    df_pw = df_pw[df_pw.index != ""]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_pw) * 0.35)))

    y = range(len(df_pw))

    # Stacked bars: with SMILES (colored) + without (gray)
    bars_with = ax.barh(
        y, df_pw["with_smiles"], color="#2ecc71", edgecolor="white",
        label="With SMILES", height=0.7,
    )
    bars_without = ax.barh(
        y, df_pw["without_smiles"], left=df_pw["with_smiles"],
        color="#bdc3c7", edgecolor="white",
        label="Without SMILES", height=0.7,
    )

    # Annotate with counts
    for i, (_, row) in enumerate(df_pw.iterrows()):
        total = row["total"]
        with_s = row["with_smiles"]
        ax.text(
            total + 0.3, i,
            f"{with_s}/{total}",
            va="center", fontsize=8,
        )

    ax.set_yticks(list(y))
    ax.set_yticklabels(df_pw.index, fontsize=9)
    ax.set_xlabel("Number of Drugs", fontsize=11)
    ax.set_title(
        "SMILES Coverage by Target Pathway",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    path = fig_dir / "drug_pathway_coverage.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# HELPER: Build mol objects + bitInfo for visualization
# =============================================================================

def _build_mol_map(
    df_smiles: pd.DataFrame,
    ecfp_names: list,
) -> dict:
    """Parse SMILES into RDKit mol objects with fingerprint bitInfo.

    Parameters
    ----------
    df_smiles : pd.DataFrame
        Drug names and SMILES strings.
    ecfp_names : list
        Drug names that have valid fingerprints.

    Returns
    -------
    dict
        {drug_name: {"mol": Mol, "smiles": str, "bitInfo": dict, "fp": BitVect}}
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    # Suppress noisy deprecation warnings from DrawMorganBit internals
    RDLogger.DisableLog("rdApp.warning")

    smiles_map = dict(zip(df_smiles["drug_name"].str.strip(), df_smiles["smiles"]))
    mol_map = {}

    for name in ecfp_names:
        smi = smiles_map.get(name)
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=ECFP_RADIUS, nBits=ECFP_NBITS, bitInfo=bi
        )
        mol_map[name] = {"mol": mol, "smiles": smi, "bitInfo": bi, "fp": fp}

    RDLogger.EnableLog("rdApp.warning")
    return mol_map


# =============================================================================
# FIGURE 5: Same-Pathway vs Cross-Pathway Tanimoto Distributions
# =============================================================================

def plot_tanimoto_distributions(
    arr_ecfp: np.ndarray,
    ecfp_names: list,
    df_index: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Plot overlapping histograms of same-pathway vs cross-pathway Tanimoto.

    Directly demonstrates that the ECFP encoding produces higher similarity
    scores for drugs sharing a target pathway than for unrelated drugs.

    Parameters
    ----------
    arr_ecfp : np.ndarray
        ECFP fingerprints (N, 2048).
    ecfp_names : list
        Drug names for each row.
    df_index : pd.DataFrame
        Drug metadata with pathway info.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    _set_style()

    logger.info("  Computing same-pathway vs cross-pathway Tanimoto distributions...")
    sim = compute_tanimoto_matrix(arr_ecfp)

    # Map names to pathways
    name_to_pw = dict(zip(df_index["drug_name"], df_index["pathway"]))
    pathways = [name_to_pw.get(n, "") for n in ecfp_names]

    # Collect upper-triangle pairwise values
    same_pw = []
    diff_pw = []
    n = len(ecfp_names)
    for i in range(n):
        for j in range(i + 1, n):
            if not pathways[i] or not pathways[j]:
                continue
            val = float(sim[i, j])
            if pathways[i] == pathways[j]:
                same_pw.append(val)
            else:
                diff_pw.append(val)

    if not same_pw or not diff_pw:
        logger.warning("  Insufficient pathway data for distribution plot, skipping")
        return

    same_pw = np.array(same_pw)
    diff_pw = np.array(diff_pw)

    # Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(same_pw, diff_pw, alternative="greater")

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 51)
    ax.hist(diff_pw, bins=bins, alpha=0.6, color="#3498db", density=True,
            label=f"Different pathway (n={len(diff_pw):,})", edgecolor="white")
    ax.hist(same_pw, bins=bins, alpha=0.6, color="#e74c3c", density=True,
            label=f"Same pathway (n={len(same_pw):,})", edgecolor="white")

    # Median lines
    ax.axvline(np.median(diff_pw), color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"Median diff: {np.median(diff_pw):.3f}")
    ax.axvline(np.median(same_pw), color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Median same: {np.median(same_pw):.3f}")

    # Annotate p-value
    p_str = f"p < 1e-{int(-np.log10(p_val))}" if p_val < 0.001 else f"p = {p_val:.4f}"
    ax.text(0.97, 0.95, f"Mann-Whitney U\n{p_str}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    ax.set_xlabel("Tanimoto Similarity (ECFP-4)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "ECFP-4 Encodes Pathway Relationships:\nSame-Pathway Drugs Are More Similar",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper center")

    plt.tight_layout()
    path = fig_dir / "tanimoto_same_vs_diff_pathway.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 6: Similarity Map Drug Pairs
# =============================================================================

def _mol_to_np_image(mol, size: int = 400, highlight_atoms: list = None):
    """Render an RDKit mol to a numpy RGB array (headless-safe).

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule to render.
    size : int
        Image width/height in pixels.
    highlight_atoms : list or None
        Atom indices to highlight in green.

    Returns
    -------
    np.ndarray
        RGB image array of shape (size, size, 3).
    """
    from rdkit.Chem import AllChem, Draw

    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(
        mol, size=(size, size),
        highlightAtoms=highlight_atoms or [],
    )
    return np.array(img)


def _get_shared_bit_atoms(mol_a, mol_b, bi_a: dict, bi_b: dict):
    """Find atoms involved in fingerprint bits shared between two molecules.

    Returns
    -------
    tuple of (list, list)
        Highlight atom indices for mol_a and mol_b.
    """
    shared_bits = set(bi_a.keys()) & set(bi_b.keys())
    atoms_a = set()
    atoms_b = set()
    for bit in shared_bits:
        for center, rad in bi_a[bit]:
            atoms_a.add(center)
        for center, rad in bi_b[bit]:
            atoms_b.add(center)
    return sorted(atoms_a), sorted(atoms_b)


def plot_similarity_map_pairs(
    df_smiles: pd.DataFrame,
    arr_ecfp: np.ndarray,
    ecfp_names: list,
    df_index: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Plot drug pair comparisons: high-similarity vs low-similarity.

    Renders molecule structures side by side with shared-bit atoms
    highlighted in green, demonstrating that same-pathway drugs share
    molecular substructures while different-pathway drugs do not.

    Parameters
    ----------
    df_smiles : pd.DataFrame
        Drug SMILES data.
    arr_ecfp : np.ndarray
        ECFP fingerprints (N, 2048).
    ecfp_names : list
        Drug names for each row.
    df_index : pd.DataFrame
        Drug metadata with pathway info.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logger.warning("  RDKit not available, skipping similarity map pairs")
        return

    _set_style()

    logger.info("  Generating similarity map drug pairs...")

    mol_map = _build_mol_map(df_smiles, ecfp_names)
    sim = compute_tanimoto_matrix(arr_ecfp)
    name_to_pw = dict(zip(df_index["drug_name"], df_index["pathway"]))

    # Filter mol_map to exclude simple salts/ions (< 5 heavy atoms)
    def _is_real_molecule(info: dict) -> bool:
        return info["mol"].GetNumHeavyAtoms() >= 5

    # Find best same-pathway pair (high sim) and best diff-pathway pair (low sim)
    best_same = (-1, -1, 0.0)
    best_diff = (-1, -1, 1.0)
    n = len(ecfp_names)

    for i in range(n):
        pw_i = name_to_pw.get(ecfp_names[i], "")
        info_i = mol_map.get(ecfp_names[i])
        if not pw_i or not info_i or not _is_real_molecule(info_i):
            continue
        for j in range(i + 1, n):
            pw_j = name_to_pw.get(ecfp_names[j], "")
            info_j = mol_map.get(ecfp_names[j])
            if not pw_j or not info_j or not _is_real_molecule(info_j):
                continue
            val = float(sim[i, j])
            if pw_i == pw_j and val > best_same[2] and val < 0.99:
                best_same = (i, j, val)
            if pw_i != pw_j and val < best_diff[2] and val > 0.01:
                best_diff = (i, j, val)

    pairs = []
    if best_same[0] >= 0:
        i, j, s = best_same
        pairs.append((ecfp_names[i], ecfp_names[j], s, "Same pathway"))
    if best_diff[0] >= 0:
        i, j, s = best_diff
        pairs.append((ecfp_names[i], ecfp_names[j], s, "Different pathway"))

    if not pairs:
        logger.warning("  Could not find suitable drug pairs, skipping")
        return

    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 5.5 * len(pairs)))
    if len(pairs) == 1:
        axes = axes.reshape(1, 2)

    for row, (name_a, name_b, tani, label) in enumerate(pairs):
        info_a = mol_map.get(name_a)
        info_b = mol_map.get(name_b)
        if not info_a or not info_b:
            continue

        # Find atoms involved in shared fingerprint bits
        hl_a, hl_b = _get_shared_bit_atoms(
            info_a["mol"], info_b["mol"],
            info_a["bitInfo"], info_b["bitInfo"],
        )

        pw_a = name_to_pw.get(name_a, "?")
        pw_b = name_to_pw.get(name_b, "?")

        n_shared = len(set(info_a["bitInfo"].keys()) & set(info_b["bitInfo"].keys()))
        n_total_a = len(info_a["bitInfo"])
        n_total_b = len(info_b["bitInfo"])

        for col, (info, name, pw, hl) in enumerate([
            (info_a, name_a, pw_a, hl_a),
            (info_b, name_b, pw_b, hl_b),
        ]):
            ax = axes[row, col]
            try:
                img_arr = _mol_to_np_image(info["mol"], size=500,
                                           highlight_atoms=hl)
                ax.imshow(img_arr)
            except Exception as e:
                logger.warning(f"  MolToImage failed for {name}: {e}")
                ax.text(0.5, 0.5, f"{name}\n(rendering failed)",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            ax.set_title(f"{name}\n({pw})", fontsize=11, fontweight="bold")
            ax.axis("off")

        # Annotate the row
        axes[row, 0].text(
            -0.02, 0.5,
            f"{label}\nTanimoto = {tani:.3f}\n{n_shared} shared bits",
            transform=axes[row, 0].transAxes,
            fontsize=11, fontweight="bold", rotation=90,
            va="center", ha="right",
        )

    plt.suptitle(
        "Drug Pair Comparison (ECFP-4)\n"
        "Highlighted atoms participate in shared fingerprint bits",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = fig_dir / "drug_similarity_maps.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 7: Shared-Bit Substructure Examples (DrawMorganBit)
# =============================================================================

def plot_shared_bits(
    df_smiles: pd.DataFrame,
    ecfp_names: list,
    df_index: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """Show molecular substructures for fingerprint bits shared by same-pathway drugs.

    Demonstrates WHY similar drugs produce similar fingerprints by rendering
    the specific chemical fragments that activate shared bits.

    Parameters
    ----------
    df_smiles : pd.DataFrame
        Drug SMILES data.
    ecfp_names : list
        Drug names for each row.
    df_index : pd.DataFrame
        Drug metadata with pathway info.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
    except ImportError:
        logger.warning("  RDKit Draw not available, skipping shared bits plot")
        return

    _set_style()

    logger.info("  Finding shared substructure bits for same-pathway drugs...")

    mol_map = _build_mol_map(df_smiles, ecfp_names)
    name_to_pw = dict(zip(df_index["drug_name"], df_index["pathway"]))

    # Find the largest real pathway with >= 2 drugs in mol_map
    skip_words = {"other", "unknown", "nan", "unclassified"}
    pw_counts = {}
    for name in ecfp_names:
        pw = name_to_pw.get(name, "")
        # Skip empty or generic pathway labels (e.g. "Other, kinases")
        if not pw or any(w in pw.lower() for w in skip_words):
            continue
        if name in mol_map:
            pw_counts.setdefault(pw, []).append(name)

    # Pick the pathway with most drugs
    best_pw = max(pw_counts, key=lambda k: len(pw_counts[k]), default=None)
    if best_pw is None or len(pw_counts[best_pw]) < 2:
        logger.warning("  Insufficient pathway data for shared bits plot, skipping")
        return

    pw_drugs = pw_counts[best_pw][:6]  # up to 6 drugs from the pathway
    logger.info(f"  Using pathway '{best_pw}' with {len(pw_drugs)} drugs")

    # Find bits shared by >= 2 drugs in this pathway
    bit_drug_map = {}  # bit_id -> list of drug names that have it
    for name in pw_drugs:
        info = mol_map[name]
        for bit_id in info["bitInfo"]:
            bit_drug_map.setdefault(bit_id, []).append(name)

    # Sort by number of drugs sharing the bit (most shared first)
    shared_bits = [(bid, drugs) for bid, drugs in bit_drug_map.items()
                   if len(drugs) >= 2]
    shared_bits.sort(key=lambda x: -len(x[1]))

    if not shared_bits:
        logger.warning("  No shared bits found, skipping")
        return

    # Pick up to 4 shared bits, each shown across 2 different drugs
    # Prefer bits shared by many drugs, and pick diverse drug pairs
    display_bits = shared_bits[:4]

    n_bits = len(display_bits)

    # Layout: n_bits rows x 2 columns (each row = one shared bit shown on 2 drugs)
    fig, axes = plt.subplots(n_bits, 2, figsize=(10, 5 * n_bits))
    if n_bits == 1:
        axes = axes.reshape(1, 2)

    def _get_bit_highlight_atoms(mol, bi, bit_id):
        """Get atom indices for a specific fingerprint bit."""
        atoms = []
        radius = 0
        if bit_id in bi:
            for center, rad in bi[bit_id]:
                atoms.append(center)
                radius = rad
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms.append(bond.GetBeginAtomIdx())
                    atoms.append(bond.GetEndAtomIdx())
        return sorted(set(atoms)), radius

    for row_idx, (bit_id, drugs) in enumerate(display_bits):
        # Pick two DIFFERENT drugs that share this bit
        drug_a = drugs[0]
        drug_b = drugs[1] if len(drugs) > 1 else drugs[0]

        for col_idx, drug_name in enumerate([drug_a, drug_b]):
            ax = axes[row_idx, col_idx]
            info = mol_map[drug_name]
            mol = info["mol"]
            bi = info["bitInfo"]

            highlight_atoms, radius = _get_bit_highlight_atoms(mol, bi, bit_id)

            try:
                img_arr = _mol_to_np_image(mol, size=400,
                                           highlight_atoms=highlight_atoms)
                ax.imshow(img_arr)
                ax.set_title(
                    f"{drug_name[:30]}",
                    fontsize=9, fontweight="bold",
                )
            except Exception as e:
                logger.warning(f"  MolToImage failed for {drug_name} bit {bit_id}: {e}")
                ax.text(0.5, 0.5, f"{drug_name}\n(render failed)",
                        ha="center", va="center", transform=ax.transAxes)

            ax.axis("off")

        # Row label on the left side
        axes[row_idx, 0].text(
            -0.05, 0.5,
            f"Bit {bit_id} (r={radius})\n"
            f"Shared by {len(drugs)}/{len(pw_drugs)} drugs",
            transform=axes[row_idx, 0].transAxes,
            fontsize=9, fontweight="bold", rotation=90,
            va="center", ha="right",
        )

    plt.suptitle(
        f"Shared ECFP-4 Bits Across Drugs: {best_pw}\n"
        f"Each row shows one fingerprint bit highlighted on two different drugs",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = fig_dir / "drug_shared_bits.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 8: Bit Frequency & Density Analysis
# =============================================================================

def plot_bit_statistics(
    arr_ecfp: np.ndarray,
    fig_dir: Path,
) -> None:
    """Plot fingerprint bit density and frequency statistics.

    Shows the information content and sparsity of the ECFP encoding.

    Parameters
    ----------
    arr_ecfp : np.ndarray
        ECFP fingerprints (N, 2048).
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt

    _set_style()

    logger.info("  Computing bit statistics...")

    bits_per_mol = arr_ecfp.sum(axis=1)
    bit_freq = arr_ecfp.sum(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel (a): Bits on per molecule ---
    ax = axes[0]
    ax.hist(bits_per_mol, bins=30, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(bits_per_mol), color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(bits_per_mol):.0f}")
    ax.set_xlabel("Active Bits per Drug", fontsize=11)
    ax.set_ylabel("Number of Drugs", fontsize=11)
    ax.set_title("(a) Fingerprint Density", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.text(0.95, 0.85,
            f"Sparsity: {1 - np.mean(bits_per_mol)/ECFP_NBITS:.1%}\n"
            f"of {ECFP_NBITS} bits",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- Panel (b): Top 30 most frequent bits ---
    ax = axes[1]
    top_n = 30
    sorted_idx = np.argsort(bit_freq)[::-1][:top_n]
    ax.bar(range(top_n), bit_freq[sorted_idx], color="#3498db", edgecolor="white")
    ax.set_xlabel("Bit Rank", fontsize=11)
    ax.set_ylabel("Number of Drugs with Bit Active", fontsize=11)
    ax.set_title(f"(b) Top {top_n} Most Frequent Bits", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, top_n, 5))

    # --- Panel (c): Bit frequency distribution ---
    ax = axes[2]
    # Exclude always-zero bits
    nonzero_freq = bit_freq[bit_freq > 0]
    ax.hist(nonzero_freq, bins=40, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of Drugs Activating Bit", fontsize=11)
    ax.set_ylabel("Number of Bits", fontsize=11)
    ax.set_title("(c) Bit Frequency Distribution", fontsize=12, fontweight="bold")

    n_zero = int((bit_freq == 0).sum())
    n_nonzero = int((bit_freq > 0).sum())
    ax.text(0.95, 0.85,
            f"Active bits: {n_nonzero}/{ECFP_NBITS}\n"
            f"Unused bits: {n_zero}",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(
        "ECFP-4 Fingerprint Statistics (2048-bit Morgan, radius=2)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = fig_dir / "ecfp_bit_statistics.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 9: UMAP + HDBSCAN Clustering Analysis
# =============================================================================

# Broad pathway superclasses for cleaner annotation
_PATHWAY_SUPERCLASS = {
    "ERK MAPK signaling": "Kinase signaling",
    "PI3K/MTOR signaling": "Kinase signaling",
    "RTK signaling": "Kinase signaling",
    "ABL signaling": "Kinase signaling",
    "JNK and p38 signaling": "Kinase signaling",
    "IGF1R signaling": "Kinase signaling",
    "WNT signaling": "Developmental",
    "Hedgehog signaling": "Developmental",
    "DNA replication": "DNA damage / replication",
    "Genome integrity": "DNA damage / replication",
    "Mitosis": "Cell cycle / mitosis",
    "Cell cycle": "Cell cycle / mitosis",
    "Apoptosis regulation": "Apoptosis",
    "Chromatin histone acetylation": "Epigenetic",
    "Chromatin histone methylation": "Epigenetic",
    "Chromatin other": "Epigenetic",
    "Cytoskeleton": "Cytoskeleton",
    "Metabolism": "Metabolism",
    "Hormone-related": "Hormone-related",
    "p53 pathway": "p53 / apoptosis",
}


def _superclass(pathway: str) -> str:
    """Map a TARGET_PATHWAY to a broad superclass."""
    return _PATHWAY_SUPERCLASS.get(pathway, "Other")


def plot_umap_hdbscan(
    arr_ecfp: np.ndarray,
    ecfp_names: list[str],
    df_index: pd.DataFrame,
    fig_dir: Path,
) -> None:
    """UMAP embedding of Tanimoto distances with HDBSCAN clustering.

    Produces:
      - A 2-panel figure: (a) colored by HDBSCAN cluster, (b) colored by
        pathway superclass for comparison
      - A CSV table of cluster composition and pathway enrichment

    Parameters
    ----------
    arr_ecfp : np.ndarray
        ECFP fingerprints (N, 2048).
    ecfp_names : list of str
        Drug names for each row.
    df_index : pd.DataFrame
        Drug metadata with pathway info.
    fig_dir : Path
        Output directory for figures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    try:
        import umap
    except ImportError:
        logger.warning("  umap-learn not installed, skipping UMAP clustering plot")
        logger.warning("  Install with: pip install umap-learn")
        return

    try:
        import hdbscan
    except ImportError:
        logger.warning("  hdbscan not installed, skipping UMAP clustering plot")
        logger.warning("  Install with: pip install hdbscan")
        return

    _set_style()

    logger.info("  Computing Tanimoto distance matrix for UMAP...")
    sim = compute_tanimoto_matrix(arr_ecfp)
    dist = (1.0 - sim).astype(np.float64)  # HDBSCAN requires float64

    # --- UMAP on precomputed distance ---
    logger.info("  Running UMAP (metric=precomputed)...")
    reducer = umap.UMAP(
        metric="precomputed",
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(dist)

    # --- HDBSCAN clustering ---
    logger.info("  Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric="precomputed",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(dist)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    logger.info(f"  HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

    # --- Pathway superclass mapping ---
    name_to_pw = dict(zip(df_index["drug_name"], df_index["pathway"]))
    superclasses = [_superclass(name_to_pw.get(n, "")) for n in ecfp_names]
    unique_super = sorted(set(superclasses))
    if "Other" in unique_super:
        unique_super.remove("Other")
        unique_super.append("Other")

    # Color palettes
    cluster_ids = sorted(set(labels) - {-1})
    n_ccolors = max(len(cluster_ids), 1)
    cluster_cmap = plt.cm.tab20 if n_ccolors <= 20 else plt.cm.gist_ncar
    cluster_colors = {
        cid: cluster_cmap(i / max(n_ccolors, 1))
        for i, cid in enumerate(cluster_ids)
    }
    cluster_colors[-1] = "#cccccc"  # noise = gray

    n_scolors = len(unique_super)
    super_cmap = plt.cm.Set2 if n_scolors <= 8 else plt.cm.tab20
    super_colors = {}
    for i, sc in enumerate(unique_super):
        if sc == "Other":
            super_colors[sc] = "#cccccc"
        else:
            super_colors[sc] = super_cmap(i / max(n_scolors, 1))

    # --- Two-panel figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Panel (a): HDBSCAN clusters
    for cid in [-1] + cluster_ids:
        mask = labels == cid
        if not mask.any():
            continue
        lbl = f"Cluster {cid} ({mask.sum()})" if cid >= 0 else f"Noise ({mask.sum()})"
        ax1.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cluster_colors[cid]],
            label=lbl,
            s=45 if cid >= 0 else 20,
            alpha=0.8 if cid >= 0 else 0.4,
            edgecolors="white", linewidth=0.3,
            marker="o" if cid >= 0 else "x",
        )

    ax1.set_xlabel("UMAP 1", fontsize=12)
    ax1.set_ylabel("UMAP 2", fontsize=12)
    ax1.set_title("(a) HDBSCAN Chemical Clusters", fontsize=13, fontweight="bold")
    ax1.legend(
        bbox_to_anchor=(0.0, -0.15), loc="upper left",
        fontsize=7, frameon=True, ncol=3,
        title="Cluster ID", title_fontsize=8,
    )

    # Panel (b): Pathway superclass overlay
    for sc in unique_super:
        mask = np.array([s == sc for s in superclasses])
        if not mask.any():
            continue
        ax2.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[super_colors[sc]],
            label=f"{sc} ({mask.sum()})",
            s=45, alpha=0.75,
            edgecolors="white", linewidth=0.3,
        )

    ax2.set_xlabel("UMAP 1", fontsize=12)
    ax2.set_ylabel("UMAP 2", fontsize=12)
    ax2.set_title("(b) Pathway Superclass Overlay", fontsize=13, fontweight="bold")
    ax2.legend(
        bbox_to_anchor=(0.0, -0.15), loc="upper left",
        fontsize=8, frameon=True, ncol=2,
        title="Pathway Superclass", title_fontsize=9,
    )

    plt.suptitle(
        "Drug Chemical Space: UMAP on Tanimoto Distance + HDBSCAN Clustering\n"
        "ECFP-4 fingerprints reveal natural chemical groupings that align with "
        "target biology",
        fontsize=13, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    path = fig_dir / "drug_umap_hdbscan.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {path}")

    # --- Cluster enrichment table ---
    _save_cluster_enrichment(
        labels, ecfp_names, name_to_pw, superclasses, fig_dir
    )


def _save_cluster_enrichment(
    labels: np.ndarray,
    ecfp_names: list[str],
    name_to_pw: dict[str, str],
    superclasses: list[str],
    fig_dir: Path,
) -> None:
    """Save cluster composition and pathway enrichment to CSV.

    For each cluster, reports the top pathways, superclasses, and member drugs.

    Parameters
    ----------
    labels : np.ndarray
        HDBSCAN cluster labels (-1 = noise).
    ecfp_names : list of str
        Drug names.
    name_to_pw : dict
        Drug name to pathway mapping.
    superclasses : list of str
        Superclass labels per drug.
    fig_dir : Path
        Output directory.
    """
    from collections import Counter

    records = []
    for cid in sorted(set(labels)):
        mask = labels == cid
        members = [ecfp_names[i] for i in range(len(ecfp_names)) if mask[i]]
        pathways = [name_to_pw.get(n, "") for n in members]
        supers = [superclasses[i] for i in range(len(superclasses)) if mask[i]]

        pw_counts = Counter(p for p in pathways if p)
        sc_counts = Counter(s for s in supers if s and s != "Other")

        top_pw = pw_counts.most_common(3)
        top_sc = sc_counts.most_common(2)

        records.append({
            "cluster": int(cid),
            "n_drugs": len(members),
            "top_pathway_1": f"{top_pw[0][0]} ({top_pw[0][1]})" if len(top_pw) > 0 else "",
            "top_pathway_2": f"{top_pw[1][0]} ({top_pw[1][1]})" if len(top_pw) > 1 else "",
            "top_pathway_3": f"{top_pw[2][0]} ({top_pw[2][1]})" if len(top_pw) > 2 else "",
            "top_superclass": f"{top_sc[0][0]} ({top_sc[0][1]})" if len(top_sc) > 0 else "",
            "drugs": "; ".join(sorted(members)[:10]) + ("..." if len(members) > 10 else ""),
        })

    df_enrich = pd.DataFrame(records)
    path = fig_dir / "cluster_enrichment.csv"
    df_enrich.to_csv(path, index=False)
    logger.info(f"  Saved {path}")

    # Print summary to log
    logger.info("  Cluster enrichment summary:")
    for _, row in df_enrich.iterrows():
        cid = row["cluster"]
        label = "Noise" if cid == -1 else f"Cluster {cid}"
        logger.info(
            f"    {label} ({row['n_drugs']} drugs): "
            f"{row['top_superclass']} | {row['top_pathway_1']}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Orchestrate drug encoding and figure generation."""
    args = parse_args()
    data_dir = args.data_dir.resolve()

    setup_logging(data_dir / "encode_drugs.log")

    logger.info("=" * 70)
    logger.info("DRUG ENCODING + VISUALIZATION")
    logger.info("=" * 70)
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  ECFP settings: radius={ECFP_RADIUS}, bits={ECFP_NBITS}")

    # ---- Load inputs ----
    logger.info("")
    logger.info("[1/6] Loading input files...")
    df_smiles = load_smiles(data_dir)
    df_compounds = load_compounds(data_dir)
    df_avail = load_available_drugs(data_dir)
    df_sample_meta = load_sample_metadata(data_dir)

    # ---- Match and report coverage ----
    logger.info("")
    logger.info("[2/6] Matching drugs and reporting coverage...")
    df_index = match_drugs(df_avail, df_smiles, df_compounds)

    # ---- Generate ECFP-4 fingerprints ----
    logger.info("")
    logger.info("[3/6] Generating ECFP-4 fingerprints...")
    arr_ecfp, ecfp_names = generate_ecfp(df_smiles, df_index)

    if arr_ecfp.shape[0] == 0:
        logger.error("  No valid fingerprints generated. Aborting.")
        sys.exit(1)

    # ---- Generate one-hot encoding ----
    logger.info("")
    logger.info("[4/6] Generating one-hot encoding...")
    arr_onehot = generate_onehot(len(df_avail))

    # ---- Save encodings ----
    logger.info("")
    logger.info("[5/6] Saving encodings...")
    embed_dir = data_dir.parent / "embeddings"
    save_encodings(arr_ecfp, arr_onehot, ecfp_names, df_index, embed_dir)

    # ---- Generate figures ----
    if args.no_figures:
        logger.info("")
        logger.info("[6/6] Skipping figures (--no-figures)")
    else:
        logger.info("")
        logger.info("[6/6] Generating figures...")
        fig_dir = data_dir.parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Original 4 figures
        plot_tsne(arr_ecfp, ecfp_names, df_index, fig_dir)
        plot_similarity_heatmap(arr_ecfp, ecfp_names, df_index, fig_dir)
        plot_dashboard(df_index, df_compounds, df_avail, df_sample_meta, fig_dir)
        plot_pathway_coverage(df_index, fig_dir)

        # New figures: demonstrate encoding quality
        plot_tanimoto_distributions(arr_ecfp, ecfp_names, df_index, fig_dir)
        plot_similarity_map_pairs(
            df_smiles, arr_ecfp, ecfp_names, df_index, fig_dir
        )
        plot_shared_bits(df_smiles, ecfp_names, df_index, fig_dir)
        plot_bit_statistics(arr_ecfp, fig_dir)

        # UMAP + HDBSCAN clustering analysis
        plot_umap_hdbscan(arr_ecfp, ecfp_names, df_index, fig_dir)

    # ---- Summary ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Modeled drugs:     {len(df_index)}")
    logger.info(f"  With SMILES:       {df_index['has_smiles'].sum()}")
    logger.info(f"  ECFP fingerprints: {arr_ecfp.shape}")
    logger.info(f"  One-hot encoding:  {arr_onehot.shape}")
    logger.info(f"  Embeddings dir:    {embed_dir}")
    if not args.no_figures:
        logger.info(f"  Figures dir:       {fig_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
