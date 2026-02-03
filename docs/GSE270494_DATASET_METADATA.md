# GSE270494 Dataset Metadata

## Dataset Overview

**Title**: DNA Methylation Database of Human and Mouse Hematological Malignancy Cell Lines

**GEO Accession**: GSE270494

**Publication**: Noguera-Castells et al., *Leukemia* (2025)

**Purpose**: Comprehensive DNA methylation profiling of hematological malignancy cell lines to identify pharmacoepigenetic biomarkers and understand methylation patterns in leukemia, lymphoma, and multiple myeloma.

**Data Type**: DNA Methylation (Illumina Infinium HumanMethylation450K BeadChip)

---

## Dataset Dimensions

### Raw Data
- **CpG Sites (Features)**: 760,090
- **Samples (Observations)**: 180 human cell lines (30 additional mouse cell lines available but not yet analyzed)
- **Total Data Points**: 136,816,200 methylation measurements
- **File Size**: 1.1 GB (compressed)
- **Format**: CSV.GZ (tab-separated beta-values with detection p-values)

### Processed Data
- **Top Variable CpG Sites**: 10,000 (variance range: 0.1362 - 0.2058)
- **Annotated Samples**: 71 of 180 (39% with disease classification)
- **Unknown Samples**: 109 of 180 (61% lacking disease annotation)

---

## Observations (Rows in Transposed Data)

### Sample Characteristics

**Total Samples**: 180 human hematological malignancy cell lines

**Sample Type**: Immortalized cancer cell lines (in vitro models)

**Sample ID Format**: Cell line names (e.g., HEL, K-562, MOLT-4, Jurkat, Raji)

### Disease Type Distribution (Target Variable)

| Disease Type | Count | Percentage | Full Name |
|--------------|-------|------------|-----------|
| AML | 21 | 11.7% | Acute Myeloid Leukemia |
| DLBCL | 11 | 6.1% | Diffuse Large B-cell Lymphoma |
| B-ALL | 8 | 4.4% | B-cell Acute Lymphoblastic Leukemia |
| T-ALL | 7 | 3.9% | T-cell Acute Lymphoblastic Leukemia |
| MM | 6 | 3.3% | Multiple Myeloma |
| HL | 6 | 3.3% | Hodgkin Lymphoma |
| MCL | 4 | 2.2% | Mantle Cell Lymphoma |
| BL | 3 | 1.7% | Burkitt Lymphoma |
| CML | 3 | 1.7% | Chronic Myeloid Leukemia |
| TCL | 2 | 1.1% | T-cell Lymphoma |
| Unknown | 109 | 60.6% | Not yet annotated |

**Data Source**: Literature-based cell line disease mapping

**Annotation Status**: Partial (requires paper's supplementary Dataset S1 for complete annotations)

---

## Features (Columns)

### Primary Features: DNA Methylation Beta-Values

**Feature Type**: Continuous numerical values

**Range**: [0.0, 1.0]
- 0.0 = Completely unmethylated
- 1.0 = Completely methylated

**Actual Observed Range**: [0.0048, 0.9953]

**Feature IDs**: CpG site identifiers (e.g., cg00000029, cg00000109, cg00000155)

**ID Format**: `cg########` (Illumina 450K array probe IDs)

**Total Features**: 760,090 CpG sites across the genome

### Feature Categories by Variability

| Category | Count | Variance Threshold | Biological Significance |
|----------|-------|-------------------|------------------------|
| High Variance | 10,000 | > 0.1362 | Potential disease biomarkers |
| Medium Variance | ~85,384 | 0.05 - 0.1362 | Moderate differential methylation |
| Low Variance | ~664,706 | < 0.05 | Constitutively methylated/unmethylated |

### Secondary Features: Detection P-Values

**Description**: Statistical confidence of methylation detection

**Format**: P-values ranging [0, 1]

**Usage**: Quality control (filtered out in analysis)

**Status**: Removed from final analysis dataset

---

## Derived Features

### Methylation Categories

Based on beta-value thresholds:
- **Hypomethylated** (β < 0.3): 31.2% of measurements
- **Intermediate** (0.3 ≤ β ≤ 0.7): 16.3% of measurements
- **Hypermethylated** (β > 0.7): 52.5% of measurements

### Principal Components (PCA)

**File**: `data/processed/GSE270494_PCA_results.csv`

**Input**: Top 10,000 most variable CpG sites (standardized)

**Components**: 10 principal components (PC1-PC10)

**Explained Variance**:
- PC1: 29.23%
- PC2: 15.89%
- PC3: 7.37%
- PC4: 3.12%
- PC5: 2.24%
- PC6-PC10: 6.77% (cumulative)
- **Total (PC1-PC10)**: 64.62%

**Use Cases**:
- Dimensionality reduction
- Disease clustering visualization
- Batch effect detection
- Sample similarity analysis

---

## Target Variables

### Primary Target: Disease Type

**Variable Name**: `disease_type`

**Type**: Categorical (multi-class)

**Classes**: 11 (10 disease types + Unknown)

**Distribution**: Imbalanced (AML: 21 samples, TCL: 2 samples, Unknown: 109 samples)

**Use Cases**:
- Disease classification models
- Differential methylation analysis
- Disease subtype clustering
- Biomarker discovery

### Potential Secondary Targets (Future Work)

Based on reference paper:
- **Drug Response**: Sensitivity/resistance to 186 chemotherapy drugs
- **Cell Proliferation**: Growth rates under drug treatment
- **Genomic Features**: Presence of known cancer driver mutations
- **Pathway Activation**: Enrichment scores for cancer-related pathways

---

## Data Quality Metrics

### Completeness

| Metric | Value | Status |
|--------|-------|--------|
| Missing Values | 0 (0.00%) | ✓ Excellent |
| Invalid Beta-Values | 0 (0.00%) | ✓ Excellent |
| Annotated Samples | 71/180 (39%) | ⚠ Partial |
| CpG Coverage | 760,090 sites | ✓ Comprehensive |

### Statistical Summary

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean Methylation | 0.5919 | Moderate global methylation |
| Median Methylation | 0.7619 | Bimodal distribution (expected) |
| Std Dev | ~0.35 | High variability across CpGs |
| Min Beta-Value | 0.0048 | Near-zero methylation observed |
| Max Beta-Value | 0.9953 | Near-complete methylation observed |

### Biological Validation

✓ **Beta-values in valid [0,1] range**
✓ **Bimodal distribution** (typical for DNA methylation)
✓ **Mean methylation** biologically plausible
✓ **High variance sites** enriched (95,384 CpGs with var > 0.1)
✓ **No outlier samples** detected in PCA
✓ **Disease clustering** observed in PC space (PC1+PC2)

---

## File Structure

### Raw Data Files

```
data/raw/GSE270494/
├── GSE270494_family.soft.gz (6.4 MB)
│   └── GEO metadata in SOFT format
└── GSE270494_Noguera-Castells_Average_Beta_Homo_Sapiens.csv.gz (1.1 GB)
    └── 760,090 CpG × 360 columns (180 beta + 180 detection p-values)
```

### Processed Data Files

```
data/processed/
├── GSE270494_sample_metadata.csv (2.6 KB)
│   └── 180 samples × 2 columns (sample_name, disease_type)
├── GSE270494_PCA_results.csv (35 KB)
│   └── 180 samples × 11 columns (PC1-PC10, disease_type)
└── GSE270494_methylation_top10k_variable.csv.gz (7.0 MB)
    └── 10,000 CpG × 180 samples (filtered high-variance sites)
```

### Visualization Outputs

```
data/figures/
├── GSE270494_PCA_scatter.png (360 KB)
├── GSE270494_PCA_scree.png (184 KB)
├── GSE270494_methylation_distributions.png (380 KB)
└── GSE270494_hierarchical_clustering.png (633 KB)
```

---

## Data Access and Loading

### Python Loading Example

```python
from src.data.loading import extract_methylation_matrix
import pandas as pd

# Load full methylation matrix (760K CpG × 180 samples)
df_methylation = extract_methylation_matrix(
    'GSE270494',
    species='human',
    filter_detection_pvals=True
)

# Load sample metadata
df_metadata = pd.read_csv('data/processed/GSE270494_sample_metadata.csv',
                          index_col=0)

# Load PCA results
df_pca = pd.read_csv('data/processed/GSE270494_PCA_results.csv')

# Load top variable CpG sites only
df_top_variable = pd.read_csv(
    'data/processed/GSE270494_methylation_top10k_variable.csv.gz',
    compression='gzip',
    index_col=0
)
```

### Data Format

**Beta-Value Matrix**:
- Rows: CpG sites (features)
- Columns: Cell line samples (observations)
- Values: Methylation beta-values [0, 1]

**Metadata**:
- Rows: Cell line samples
- Columns: Annotations (disease_type)
- Index: sample_name

---

## Known Limitations

### 1. Incomplete Annotations
- **Issue**: 109/180 samples (60.6%) lack disease type annotations
- **Impact**: Reduced sample size for supervised learning
- **Mitigation**: Proceed with 71 annotated samples for disease-specific analysis
- **Resolution**: Requires supplementary Dataset S1 from publication

### 2. Class Imbalance
- **Issue**: AML (21 samples) vs TCL (2 samples) - 10.5× difference
- **Impact**: Model bias toward majority classes
- **Mitigation**: Stratified cross-validation, class weighting, SMOTE
- **Consideration**: Small sample size for rare disease types

### 3. Platform-Specific Biases
- **Issue**: Illumina 450K array has known probe biases
- **Impact**: Some CpG sites may have technical artifacts
- **Mitigation**: Use detection p-values, cross-reference with 850K EPIC array
- **Validation**: Compare findings with independent platforms

### 4. Cell Line Artifacts
- **Issue**: Cell lines may not perfectly represent primary tumors
- **Impact**: Findings may not generalize to clinical samples
- **Mitigation**: Cross-validate with primary tumor data (GSE68379, TCGA)
- **Note**: Paper validates cell line findings in primary tumors

---

## Recommended Use Cases

### ✓ Suitable For:

1. **Disease Classification**
   - Multi-class classification (11 classes)
   - Binary classification (e.g., AML vs DLBCL)
   - 71 annotated samples for supervised learning

2. **Biomarker Discovery**
   - Identify disease-specific CpG sites
   - 10,000 high-variance CpG sites pre-selected
   - Multiple testing correction required

3. **Dimensionality Reduction**
   - PCA already computed (PC1-PC10)
   - 64.62% variance explained
   - Disease clustering validated

4. **Clustering Analysis**
   - Hierarchical clustering performed
   - 71 annotated samples cluster by disease
   - Unknown samples can be assigned to clusters

5. **Comparative Analysis**
   - Compare hematological vs solid tumors (GSE68379)
   - Cross-dataset validation
   - 177 blood cancer samples in GSE68379 for overlap

### ⚠ Limitations / Cautions:

1. **Small Sample Size**: 71-180 samples (depending on annotation status)
2. **Deep Learning**: Likely underpowered for complex neural networks
3. **Rare Classes**: TCL (2 samples), BL (3 samples) - too small for reliable modeling
4. **Clinical Predictions**: Cell lines ≠ patients; validate on clinical data
5. **Batch Effects**: Single-batch data; no batch correction possible

---

## Biological Context

### DNA Methylation in Cancer

DNA methylation is an epigenetic modification where methyl groups (CH₃) are added to cytosine bases in CpG dinucleotides. In cancer:
- **Hypermethylation** of tumor suppressor promoters → gene silencing
- **Hypomethylation** of oncogenes → gene activation
- **Global patterns** distinguish cancer types

### Hematological Malignancies

These cancers arise from blood-forming tissues:
- **Leukemias**: Liquid tumors in bone marrow/blood (AML, B-ALL, T-ALL, CML)
- **Lymphomas**: Solid tumors in lymphatic system (DLBCL, HL, MCL, BL, TCL)
- **Multiple Myeloma**: Plasma cell cancer in bone marrow

### Clinical Relevance

From Noguera-Castells et al. (2025):
- **802 drug-associated differentially methylated regions (dDMRs)** identified
- **Pharmacoepigenetics**: Methylation predicts response to nucleoside analogues
- **Validation**: Cell line methylation patterns reproduce in primary tumors
- **Enrichment**: dDMRs enriched in promoters, CpG islands, DNase I hypersensitive sites
- **Cancer genes**: 5.4% of genes near dDMRs are known cancer genes (vs 3.8% background)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-20 | Initial metadata documentation |

**Last Updated**: October 20, 2025

**Contact**: See project README.md

**License**: Data from GEO (public domain); analysis code under project license

---

## References

1. Noguera-Castells A, et al. (2025). "A DNA methylation database of human and mouse hematological malignancy cell lines." *Leukemia*. GEO Accession: GSE270494.

2. Bibikova M, et al. (2011). "High density DNA methylation array with single CpG site resolution." *Genomics*, 98(4):288-295. DOI: 10.1016/j.ygeno.2011.07.007

3. Project References: `.claude/references/hematological_malignancy_paper.pdf`

---

**Notes**:
- All file paths relative to project root: `bioml_working_repo/`
- Data files gitignored (not in version control)
- Processed data can be regenerated from raw data using `src/data/loading.py`
- Random seed = 42 for all analyses (reproducibility)
