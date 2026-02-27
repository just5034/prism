# Drug Encoding Results: ECFP-4 Fingerprints for the PRISM Drug Panel

**Date**: February 26, 2026
**Author**: Generated from `encode_drugs.py` pipeline
**Dataset**: 375 modeled GDSC compounds, 314 with molecular structures

---

## Table of Contents

1. [What Is ECFP-4 Encoding?](#1-what-is-ecfp-4-encoding)
2. [Dataset Coverage](#2-dataset-coverage)
3. [Encoding Properties](#3-encoding-properties)
4. [Validation: Does the Encoding Capture Biology?](#4-validation-does-the-encoding-capture-biology)
5. [Molecular-Level Evidence](#5-molecular-level-evidence)
6. [Known Limitations](#6-known-limitations)
7. [Outputs for Downstream Models](#7-outputs-for-downstream-models)
8. [Figure Reference](#8-figure-reference)
9. [References](#9-references)

---

## 1. What Is ECFP-4 Encoding?

To use drugs as inputs to a machine learning model, we need to convert molecular structures into fixed-length numerical vectors. We use **ECFP-4** (Extended Connectivity Fingerprint, radius 2), a standard molecular fingerprint from cheminformatics.

### How it works

1. For each atom in a molecule, the algorithm examines the chemical neighborhood within 2 bonds (radius = 2, hence "ECFP-**4**" = diameter 4).
2. Each unique neighborhood is hashed into a position in a **2048-bit binary vector**.
3. If two drugs share the same chemical substructure (e.g., a morpholine ring, an aromatic amine), the same bit gets set to 1 in both vectors.

### Measuring similarity

The standard similarity metric for binary fingerprints is the **Tanimoto coefficient**:

```
Tanimoto(A, B) = |A & B| / |A | B|
```

This is the number of shared on-bits divided by the total number of on-bits across both drugs. It ranges from 0 (completely different) to 1 (identical fingerprints).

### Why ECFP-4?

- Ranked #1 across 88 virtual screening targets in the Riniker & Landrum (2013) benchmark
- Standard representation in drug response prediction (Baptista et al. 2022, Scientific Reports 2024)
- Captures local chemical substructures that relate to biological activity
- Directly compatible with GNN and deep learning pipelines

---

## 2. Dataset Coverage

### SMILES retrieval

We retrieved SMILES (molecular structure strings) from PubChem for all GDSC compounds. Of our **375 modeled drugs**, **314 (84%)** have valid SMILES and were successfully encoded. The 61 missing drugs are mostly internal screening codes without public chemical structures.

> **Figure: `dataset_overview_dashboard.png`** — Panel (c) shows the 84/16% coverage split. Panel (b) shows the distribution across 25+ target pathways. Panel (d) shows that most drugs were tested on ~870 cell lines.

### Pathway-level coverage

SMILES coverage is consistent across biological categories. Most pathways have 75-100% coverage, and several smaller pathways (ABL signaling, p53, EGFR, cytoskeleton) have 100%. The only weak spot is "Unclassified" compounds (4/10), which are poorly characterized by definition.

> **Figure: `drug_pathway_coverage.png`** — Stacked horizontal bars showing with/without SMILES for each of 26 target pathways. The encoding does not systematically miss any biological category.

---

## 3. Encoding Properties

### Fingerprint statistics

| Property | Value |
|----------|-------|
| Drugs encoded | 314 |
| Vector length | 2048 bits |
| Mean active bits per drug | 55 |
| Range | 1 - 139 |
| Sparsity | 97.3% |
| Active bit positions (used by at least 1 drug) | 1,845 / 2,048 |
| Unused bit positions | 203 |

The encoding is **highly sparse** — on average, only 55 of 2048 bits are turned on per drug. This means each drug has a distinctive "barcode." The mix of common bits (shared ubiquitous chemistry like aromatic carbons) and rare bits (unique substructures) gives the encoding both generalization ability and discrimination power.

> **Figure: `ecfp_bit_statistics.png`** — Three panels: (a) distribution of active bits per drug, (b) the 30 most common bits and how many drugs activate each, (c) the frequency distribution across all 1,845 active bits.

---

## 4. Validation: Does the Encoding Capture Biology?

The central question: **do drugs with similar fingerprints actually share biological function?** We validate this three ways, from population-level statistics to individual examples.

### 4a. Statistical test: same-pathway drugs are more similar

We computed Tanimoto similarity for all ~49,000 drug pairs and separated them into same-pathway pairs (n=3,453) and different-pathway pairs (n=45,688).

- **Mann-Whitney U test: p < 1e-16** — the difference is statistically significant
- Same-pathway median: 0.114 | Different-pathway median: 0.109
- The key signal is in the **right tail**: same-pathway pairs are far more likely to have high similarity scores (> 0.3)

The small median difference is expected and scientifically accurate — drugs targeting the same pathway can have diverse chemical scaffolds (e.g., many structurally distinct kinase inhibitors exist). The encoding captures the relationships that do exist without fabricating ones that don't.

> **Figure: `tanimoto_same_vs_diff_pathway.png`** — Overlapping histograms of same-pathway (red) vs different-pathway (blue) Tanimoto distributions, with medians and p-value annotated.

### 4b. Nearest-neighbor concordance (primary validation)

This is the standard cheminformatics validation approach (Riniker & Landrum 2013). For each drug, we find its k nearest neighbors by Tanimoto similarity and check whether they share the same target pathway.

**Key results:**

| Metric | Value |
|--------|-------|
| k=1 concordance | 29.3% |
| Random baseline | 7.3% |
| **Fold enrichment at k=1** | **4.0x** |
| k=5 concordance | 17.6% |
| **Fold enrichment at k=5** | **2.4x** |

A drug's nearest fingerprint neighbor shares its pathway **4 times more often than chance**. This enrichment is maintained across all k from 1 to 20.

**Per-pathway results (at k=5):**

| Pathway | Concordance | n drugs |
|---------|------------|---------|
| Chromatin histone acetylation | 0.43 | 14 |
| ERK MAPK signaling | 0.30 | 14 |
| PI3K/MTOR signaling | 0.28 | 33 |
| RTK signaling | 0.24 | 32 |
| EGFR signaling | 0.23 | 7 |
| Mitosis | 0.20 | 15 |
| Genome integrity | 0.18 | 11 |
| Cell cycle | 0.17 | 15 |

HDAC inhibitors (chromatin histone acetylation) are best captured because they share distinctive chemical scaffolds (hydroxamic acid groups). Kinase inhibitor pathways (ERK MAPK, PI3K/MTOR, RTK) also show strong concordance. Some pathways (chromatin histone methylation, WNT signaling) are near random, indicating chemically diverse drug sets that happen to share a target.

**Similarity-concordance relationship:**

Panel (c) shows a clear **monotonic relationship** — as Tanimoto similarity increases, the probability of sharing a pathway increases steadily. Drugs with top-1 neighbors at Tanimoto > 0.5 have 60-100% pathway match rates. Below 0.2, match rates approach random. This is the cleanest evidence that fingerprint similarity corresponds to biological relatedness.

> **Figure: `neighbor_concordance.png`** — Three panels: (a) concordance vs k with fold enrichment, (b) per-pathway concordance bar chart, (c) pathway match rate binned by Tanimoto similarity.

### 4c. Similarity heatmap

The full 314x314 Tanimoto similarity matrix, sorted by target pathway, shows visible **block structure along the diagonal**. Drugs within the same pathway tend to have higher pairwise similarity (warmer colors) than off-diagonal pairs. Some pathways show particularly tight blocks (chromatin histone acetylation, EGFR signaling).

> **Figure: `drug_similarity_heatmap.png`** — Heatmap with pathway color bar. Yellow = low similarity, dark red = high similarity.

---

## 5. Molecular-Level Evidence

### 5a. Drug pair comparison

A concrete example of what "similar" and "dissimilar" encodings look like:

**Same pathway (PI3K/MTOR signaling):**
- Temsirolimus vs Rapamycin
- Tanimoto similarity: **0.842**
- **96 shared fingerprint bits**
- Temsirolimus is a derivative of Rapamycin — the encoding correctly captures their near-identical chemistry, with almost every atom participating in shared bits

**Different pathways (DNA replication vs RTK signaling):**
- Cisplatin vs PHA-665752
- Tanimoto similarity: **0.013**
- **1 shared bit**
- Cisplatin is a simple platinum coordination compound; PHA-665752 is a complex organic molecule. The encoding correctly assigns them nearly zero similarity.

> **Figure: `drug_similarity_maps.png`** — Side-by-side molecular structures with shared-bit atoms highlighted in red. Top row: high similarity pair. Bottom row: low similarity pair.

### 5b. Shared substructure bits

Zooming into the PI3K/MTOR signaling pathway (6 drugs), we identified fingerprint bits shared by all 6 drugs and visualized which atoms activate those bits on two different drugs (AZD6482 and Pictilisib).

Both drugs share a **morpholine ring** — a well-known pharmacophore in PI3K inhibitors — and the encoding automatically identifies it as a shared feature. This demonstrates that the fingerprint bits correspond to biologically meaningful chemical fragments.

> **Figure: `drug_shared_bits.png`** — Four rows, each showing one shared fingerprint bit highlighted on two different PI3K/MTOR drugs.

---

## 6. Known Limitations

### 2D visualization is inherently limited

The t-SNE scatter plot (`drug_chemical_space_tsne.png`) appears muddled with mixed pathway colors. **This is a known mathematical limitation, not a problem with the encoding.** ECFP-4 produces 2048-bit sparse binary vectors where most drug pairs are nearly equidistant (Tanimoto ~0.1). Faithfully representing N nearly-equidistant points requires N-1 dimensions; projecting into 2D loses most neighborhood relationships (Wagen 2023). This is why we use nearest-neighbor concordance (which operates in the full 2048-dimensional space) as the primary validation.

### Coverage gaps

61 drugs (16%) lack SMILES and cannot be fingerprinted. These drugs use one-hot encoding as a fallback in the GNN pipeline. The missing drugs are distributed across pathways without systematic bias.

### Pathway labels are pharmacological, not chemical

Target pathway is a biological grouping. Drugs targeting the same pathway can have completely different chemical scaffolds (e.g., allosteric vs ATP-competitive kinase inhibitors). The encoding captures chemical similarity — the fact that it correlates with pathway labels at 4x above chance is noteworthy precisely because perfect correspondence is not expected.

### Concordance varies by pathway

Some pathways (HDAC inhibitors: 0.43 concordance) are much better captured than others (chromatin histone methylation: 0.10). This reflects genuine chemical diversity within some pharmacological classes, not an encoding failure.

---

## 7. Outputs for Downstream Models

The encoding pipeline produces the following files for the GNN drug response prediction model:

| File | Shape | Description |
|------|-------|-------------|
| `embeddings/drug_ecfp.npy` | (314, 2048) | ECFP-4 fingerprints, float32 |
| `embeddings/drug_onehot.npy` | (375, 375) | One-hot identity matrix, float32 |
| `embeddings/drug_index.csv` | 375 rows | Mapping: drug name, ECFP row index, one-hot index, has_smiles, target, pathway |

- **ECFP fingerprints** are used as drug node features in the GNN for the 314 drugs with SMILES
- **One-hot encoding** serves as a baseline comparison and fallback for the 61 drugs without SMILES
- **Drug index CSV** provides the lookup table connecting drug names to row indices in both encodings

---

## 8. Figure Reference

| # | File | Description |
|---|------|-------------|
| 1 | `drug_chemical_space_tsne.png` | t-SNE of ECFP-4 chemical space (supplementary; 2D projection has known limitations) |
| 2 | `drug_similarity_heatmap.png` | 314x314 Tanimoto similarity matrix sorted by pathway |
| 3 | `dataset_overview_dashboard.png` | Dataset scope: pathways, SMILES coverage, testing completeness |
| 4 | `drug_pathway_coverage.png` | SMILES coverage breakdown by target pathway |
| 5 | `tanimoto_same_vs_diff_pathway.png` | Same-pathway vs different-pathway Tanimoto distributions |
| 6 | `drug_similarity_maps.png` | High-similarity vs low-similarity drug pair with atom highlighting |
| 7 | `drug_shared_bits.png` | Shared fingerprint bits across PI3K/MTOR drugs |
| 8 | `ecfp_bit_statistics.png` | Fingerprint density, top bits, frequency distribution |
| 9 | `neighbor_concordance.png` | Nearest-neighbor pathway concordance (primary validation) |

---

## 9. References

- **Riniker, S. & Landrum, G.A.** (2013). Open-source platform to benchmark fingerprints for virtual screening. *J. Cheminformatics*, 5:26. — Established ECFP-4 as top-ranked fingerprint across 88 targets.
- **O'Boyle, N.M. & Sayle, R.A.** (2016). Comparing structural fingerprints using a literature-based similarity benchmark. *J. Cheminformatics*, 8:36. — Literature-validated similarity benchmarks.
- **Baptista, D. et al.** (2022). Evaluating molecular representations in machine learning models for drug response prediction. *J. Cheminformatics*, 14:63. — Benchmarked 12 molecular representations for cancer drug response.
- **Wagen, C.C.** (2023). Dimensionality reduction for molecular fingerprints. Blog post. — Explains why 2D projections of ECFP fingerprints are inherently limited.
- **Landrum, G.A.** (2021). Fingerprint similarity thresholds for database searches. RDKit Blog. — Empirical Tanimoto thresholds for molecular similarity.
- **Rogers, D. & Hahn, M.** (2010). Extended-connectivity fingerprints. *J. Chem. Inf. Model.*, 50(5):742-754. — Original ECFP paper.
