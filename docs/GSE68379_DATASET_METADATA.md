# GSE68379 Dataset Metadata

## Dataset Overview

**Title**: DNA Methylation in Human Cancer Cell Lines (Genomics of Drug Sensitivity in Cancer - GDSC)

**GEO Accession**: GSE68379

**Publication**: Garnett MJ, et al. (associated with GDSC project)

**Purpose**: Comprehensive DNA methylation profiling of cancer cell lines across 22 cancer types to identify epigenetic biomarkers of drug sensitivity and understand methylation patterns across diverse cancers.

**Data Type**: DNA Methylation (Illumina Infinium HumanMethylation450K BeadChip)

**Associated Database**: Genomics of Drug Sensitivity in Cancer (GDSC) - includes drug response data for 453 anticancer compounds

---

## Dataset Dimensions

### Raw Data
- **CpG Sites (Features)**: 485,512
- **Total Columns**: 2,056 (includes replicates and additional samples)
- **Samples with Metadata (Observations)**: 1,028 unique cancer cell lines
- **Total Data Points**: ~499,286,336 methylation measurements
- **File Size**: 3.9 GB (compressed)
- **Format**: Tab-separated text (.txt.gz)

### Processed Data
- **Top Variable CpG Sites**: 10,000 (for PCA and clustering)
- **Fully Annotated Samples**: 1,028 (100% with primary site and histology)
- **Primary Sites**: 13 organ systems
- **Primary Histologies**: 56 detailed cancer subtypes

---

## Observations (Rows in Transposed Data)

### Sample Characteristics

**Total Samples**: 1,028 human cancer cell lines

**Sample Type**: Immortalized cancer cell lines (in vitro models)

**Sample ID Formats**:
- **GSM ID**: GEO Sample IDs (e.g., GSM1669562, GSM1669563)
- **Cell Line Name**: Standard nomenclature (e.g., HeLa, A549, MCF7, K-562)
- **COSMIC ID**: Cross-referenced to COSMIC Cell Lines Database

### Primary Site Distribution (Major Target Variable)

| Primary Site | Count | Percentage | Examples |
|--------------|-------|------------|----------|
| lung | 198 | 19.3% | A549, NCI-H460, H1975 |
| blood | 177 | 17.2% | K-562, HL-60, Jurkat |
| urogenital_system | 115 | 11.2% | DU-145, PC-3, OVCAR-3 |
| digestive_system | 105 | 10.2% | HCT-116, SW480, HT-29 |
| nervous_system | 96 | 9.3% | U-87 MG, SH-SY5Y |
| skin | 62 | 6.0% | A375, SK-MEL-28 |
| breast | 48 | 4.7% | MCF7, MDA-MB-231 |
| ovary | 49 | 4.8% | OVCAR-3, SKOV3 |
| soft_tissue | 27 | 2.6% | RD, HT-1080 |
| pancreas | 20 | 1.9% | PANC-1, MIA PaCa-2 |
| bone | 18 | 1.8% | U-2 OS, Saos-2 |
| upper_aerodigestive_tract | 29 | 2.8% | Detroit 562, FaDu |
| urinary_tract | 29 | 2.8% | T24, UMRC2 |
| **Other** | 55 | 5.3% | Various rare sites |

### Primary Histology Distribution (Detailed Target Variable)

Top 20 Most Abundant Histologies:

| Histology | Count | % | Primary Sites |
|-----------|-------|---|---------------|
| lung_NSCLC_adenocarcinoma | 67 | 6.5% | lung |
| lung_small_cell_carcinoma | 64 | 6.2% | lung |
| melanoma | 56 | 5.4% | skin |
| glioma | 54 | 5.3% | nervous_system |
| breast | 52 | 5.1% | breast |
| lymphoid_neoplasm | 50+ | ~5% | blood |
| lymphoblastic_leukemia | 30+ | ~3% | blood |
| colorectal_adenocarcinoma | 40+ | ~4% | digestive_system |
| gastric_adenocarcinoma | 20+ | ~2% | digestive_system |
| ovarian_carcinoma | 40+ | ~4% | ovary |
| prostate_carcinoma | 20+ | ~2% | urogenital_system |
| bladder_carcinoma | 20+ | ~2% | urinary_tract |
| pancreatic_adenocarcinoma | 20+ | ~2% | pancreas |
| neuroblastoma | 20+ | ~2% | nervous_system |
| osteosarcoma | 15+ | ~1.5% | bone |
| ... | ... | ... | ... |
| **Total Unique**: 56 histologies |

**Data Source**: GEO metadata + COSMIC Cell Lines Database

**Annotation Status**: Complete (100% of samples annotated)

---

## Features (Columns)

### Primary Features: DNA Methylation Beta-Values

**Feature Type**: Continuous numerical values

**Range**: [0.0, 1.0]
- 0.0 = Completely unmethylated
- 1.0 = Completely methylated

**Actual Observed Range**: [0.0000, 1.0000]

**Feature IDs**: CpG site identifiers (e.g., cg00000029, cg00000109, cg00000155)

**ID Format**: `cg########` (Illumina 450K array probe IDs)

**Total Features**: 485,512 CpG sites across the genome

**Column Naming**: Originally `[CellLine]_AVG.Beta` (cleaned to `[CellLine]` for analysis)

### Feature Categories by Variability

| Category | Approx. Count | Variance Threshold | Biological Significance |
|----------|---------------|-------------------|------------------------|
| High Variance | ~10,000 | Top 10K by variance | Potential cancer biomarkers |
| Medium Variance | ~50,000 | Moderate variability | Tissue-specific methylation |
| Low Variance | ~425,512 | Minimal variation | Constitutive methylation |

### Missing Data Patterns

| Metric | Value | Status |
|--------|-------|--------|
| CpG sites with any missing values | 10,995 (2.26%) | ⚠ Minor |
| Samples with any missing values | 1,026 (99.81%) | ⚠ Sparse missingness |
| Total missing values | 47,307 (0.0095% of total) | ✓ Acceptable |

**Interpretation**: Missing values are sparse (< 0.01%) and do not affect most analyses. Imputation or filtering applied as needed.

---

## Derived Features

### Methylation Categories

Based on beta-value thresholds:
- **Hypomethylated** (β < 0.3): 30.0% of measurements
- **Intermediate** (0.3 ≤ β ≤ 0.7): 35.6% of measurements
- **Hypermethylated** (β > 0.7): 34.4% of measurements

### Principal Components (PCA)

**File**: `data/processed/GSE68379_PCA_results.csv`

**Input**: Top 10,000 most variable CpG sites (standardized)

**Components**: 10 principal components (PC1-PC10)

**Explained Variance**:
- PC1: 22.75%
- PC2: 8.87%
- PC3: 5.38%
- PC4: 4.43%
- PC5: 3.15%
- PC6: 1.72%
- PC7: 1.60%
- PC8: 1.40%
- PC9: 1.26%
- PC10: 0.86%
- **Total (PC1-PC10)**: 51.44%

**Use Cases**:
- Dimensionality reduction
- Cancer type clustering visualization
- Batch effect detection
- Cross-tissue comparison

### Histology Grouping

**Variable**: `histology_group`

**Purpose**: Simplify 56 histologies to top 10 + "Other" for visualization

**Categories**: 11 (top 10 most common + Other)

---

## Target Variables

### Primary Target: Primary Site (Organ System)

**Variable Name**: `primary site`

**Type**: Categorical (multi-class)

**Classes**: 13 organ systems

**Distribution**: Moderately balanced (lung: 198, bone: 18)

**Use Cases**:
- Tissue-of-origin classification
- Cross-tissue methylation comparison
- Pan-cancer biomarker discovery

### Secondary Target: Primary Histology (Cancer Subtype)

**Variable Name**: `primary histology`

**Type**: Categorical (multi-class, fine-grained)

**Classes**: 56 detailed cancer subtypes

**Distribution**: Highly imbalanced (NSCLC adenocarcinoma: 67, rare subtypes: 1-5)

**Use Cases**:
- Cancer subtype classification
- Histology-specific biomarkers
- Hierarchical classification (site → histology)

### Histology Group (Simplified)

**Variable Name**: `histology_group`

**Type**: Categorical (11 classes)

**Classes**: Top 10 histologies + "Other"

**Use Cases**: Visualization with manageable color palettes

### External Target: Drug Response (GDSC Database)

**Source**: Genomics of Drug Sensitivity in Cancer (GDSC)

**Format**: Separate database, linkable via cell line name / COSMIC ID

**Metrics**:
- **IC50**: Half-maximal inhibitory concentration
- **AUC**: Area under dose-response curve
- **Sensitivity/Resistance**: Binary classification

**Drugs**: 453 anticancer compounds

**Use Cases**:
- Pharmacoepigenomics
- Drug sensitivity prediction
- Methylation-response associations
- Precision oncology biomarkers

**Integration**: Requires GDSC data download (not included in this GEO dataset)

---

## Data Quality Metrics

### Completeness

| Metric | Value | Status |
|--------|-------|--------|
| Missing Values | 47,307 (0.0095%) | ✓ Minimal |
| Invalid Beta-Values | 0 (0.00%) | ✓ Excellent |
| Annotated Samples | 1,028/1,028 (100%) | ✓ Complete |
| CpG Coverage | 485,512 sites | ✓ Comprehensive |

### Statistical Summary

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean Methylation | 0.5060 | Moderate global methylation |
| Median Methylation | 0.5662 | Bimodal distribution (expected) |
| Std Dev | ~0.35 | High variability across CpGs |
| Min Beta-Value | 0.0000 | Complete unmethylation observed |
| Max Beta-Value | 1.0000 | Complete methylation observed |

### Biological Validation

✓ **Beta-values in valid [0,1] range**
✓ **Bimodal distribution** (typical for DNA methylation)
✓ **Mean methylation** biologically plausible
✓ **Missing values < 0.01%** (acceptable for large-scale arrays)
✓ **Complete annotations** (100% samples with metadata)
✓ **Cancer type clustering** observed in PCA (PC1+PC2 = 31.62%)
✓ **Tissue-specific patterns** visible in hierarchical clustering

---

## File Structure

### Raw Data Files

```
data/raw/GSE68379/
├── GSE68379_family.soft.gz (~20 MB)
│   └── GEO metadata in SOFT format
└── GSE68379_Matrix.processed.txt.gz (3.9 GB)
    └── 485,512 CpG × 2,056 columns (includes replicates)
```

### Processed Data Files

```
data/processed/
├── GSE68379_sample_metadata.csv (76 KB)
│   └── 1,028 samples × 7 columns (gsm_id, cell line, primary site, primary histology, cosmic_id, title, source_name)
├── GSE68379_PCA_results.csv (235 KB)
│   └── 1,028 samples × 14 columns (PC1-PC10, primary site, primary histology, histology_group)
└── GSE68379_top10k_variable_CpGs.csv (310 KB)
    └── 10,000 CpG IDs with variance scores
```

### Visualization Outputs

```
data/figures/
├── GSE68379_methylation_distributions.png
├── GSE68379_sample_distribution.png
├── GSE68379_PCA_scree.png
├── GSE68379_PCA_scatter.png
├── GSE68379_PCA_3D_scatter.png (interactive Plotly)
└── GSE68379_hierarchical_clustering.png
```

---

## Data Access and Loading

### Python Loading Example

```python
import pandas as pd
from pathlib import Path

# Load full methylation matrix (485K CpG × 1,028 samples)
raw_dir = Path('data/raw/GSE68379')
df_methylation_full = pd.read_csv(
    raw_dir / 'GSE68379_Matrix.processed.txt.gz',
    sep='\t',
    index_col=1,
    compression='gzip'
)

# Clean column names (remove _AVG.Beta suffix)
df_methylation_full.columns = [
    col.replace('_AVG.Beta', '')
    for col in df_methylation_full.columns
]

# Drop Row.names column if present
if 'Row.names' in df_methylation_full.columns:
    df_methylation_full = df_methylation_full.drop('Row.names', axis=1)

# Load sample metadata
df_metadata = pd.read_csv(
    'data/processed/GSE68379_sample_metadata.csv',
    index_col='source_name'
)

# Filter to samples with metadata
common_samples = list(set(df_metadata.index) & set(df_methylation_full.columns))
df_methylation = df_methylation_full[common_samples].copy()

# Load PCA results
df_pca = pd.read_csv('data/processed/GSE68379_PCA_results.csv')
```

### Data Format

**Beta-Value Matrix**:
- Rows: CpG sites (features)
- Columns: Cell line samples (observations)
- Values: Methylation beta-values [0, 1]
- Missing: NaN (sparse, <0.01%)

**Metadata**:
- Rows: Cell line samples
- Columns: 7 annotation fields
- Index: source_name (cell line identifier)

---

## Known Limitations

### 1. Sample Count Discrepancy
- **Issue**: 2,056 total columns but only 1,028 have metadata
- **Explanation**: Likely includes technical replicates or QC samples
- **Impact**: Analysis uses 1,028 well-annotated samples
- **Resolution**: Verified 100% metadata coverage for 1,028 samples

### 2. Class Imbalance (Histology)
- **Issue**: NSCLC adenocarcinoma (67) vs rare subtypes (1-5 samples)
- **Impact**: Model bias toward common cancer types
- **Mitigation**: Group rare subtypes, use class weighting, stratified CV
- **Consideration**: Hierarchical models (site → histology)

### 3. Missing Values (Sparse)
- **Issue**: 10,995 CpG sites (2.26%) have at least one missing value
- **Impact**: 47,307 total missing values (0.0095% of dataset)
- **Mitigation**: Imputation with column means or filtering affected probes
- **Quality**: Excellent overall (99.99% complete)

### 4. Platform-Specific Biases
- **Issue**: Illumina 450K array has known probe design biases
- **Impact**: SNP-affected probes, cross-reactive probes
- **Mitigation**: Filter problematic probes (Chen et al. 2013)
- **Validation**: Cross-reference with 850K EPIC array or bisulfite sequencing

### 5. Cell Line Artifacts
- **Issue**: Cell lines cultured in vitro ≠ primary tumors in vivo
- **Impact**: Findings may not generalize to clinical samples
- **Mitigation**: Validate on primary tumor datasets (TCGA)
- **Advantage**: Paired with drug response data for functional validation

### 6. Drug Response Data Separate
- **Issue**: GDSC drug response data not included in GEO dataset
- **Impact**: Pharmacoepigenomic analysis requires additional data integration
- **Resolution**: Download from GDSC website: www.cancerrxgene.org
- **Linkage**: Via cell line name or COSMIC ID

---

## Recommended Use Cases

### ✓ Suitable For:

1. **Tissue-of-Origin Classification**
   - 13 primary sites (multi-class classification)
   - 1,028 fully annotated samples
   - Moderately balanced classes (lung: 198, bone: 18)

2. **Cancer Subtype Classification**
   - 56 histologies (fine-grained classification)
   - Large sample size enables deep learning
   - Hierarchical models (site → histology)

3. **Pan-Cancer Biomarker Discovery**
   - Identify methylation patterns common across cancers
   - Tissue-specific vs universal methylation changes
   - 485K CpG sites for genome-wide screening

4. **Pharmacoepigenomics** (with GDSC data)
   - Predict drug sensitivity from methylation
   - 453 drugs × 1,028 cell lines (if GDSC data integrated)
   - Identify epigenetic resistance mechanisms

5. **Cross-Dataset Validation**
   - Compare with GSE270494 (177 blood cancers in both)
   - Validate hematological malignancy findings
   - Cross-platform consistency checks

6. **Transfer Learning**
   - Large sample size (1,028) for pre-training
   - Transfer to smaller clinical datasets
   - Feature extraction for downstream tasks

### ⚠ Limitations / Cautions:

1. **Rare Histologies**: Some subtypes have <10 samples
2. **Drug Data Separate**: Requires GDSC integration
3. **Cell Lines ≠ Patients**: Validate on clinical cohorts (TCGA)
4. **Batch Effects**: Unknown - check PCA for technical artifacts
5. **Missing Values**: Sparse but present (imputation recommended)

---

## Comparison with GSE270494

| Characteristic | GSE270494 | GSE68379 |
|----------------|-----------|----------|
| **Focus** | Hematological malignancies | Pan-cancer (22 types) |
| **Samples** | 180 | 1,028 |
| **CpG Sites** | 760,090 | 485,512 |
| **Annotations** | 39% (71/180) | 100% (1,028/1,028) |
| **Disease Types** | 10 (+ Unknown) | 13 sites, 56 histologies |
| **Missing Values** | 0 (0.00%) | 47,307 (0.0095%) |
| **Platform** | 450K | 450K |
| **File Size** | 1.1 GB | 3.9 GB |
| **Target Use** | Disease classification | Tissue/histology classification, drug response |
| **Overlap** | 177 blood cancers | 177 blood cancers |

**Synergies**:
- **Blood Cancers**: 177 shared cell line types for cross-validation
- **Platform**: Both use Illumina 450K (direct comparison)
- **Biomarkers**: Validate hematological malignancy findings across datasets
- **Drug Response**: GSE270494 (186 drugs), GSE68379 (453 drugs via GDSC)

---

## Biological Context

### Cancer Diversity

This dataset spans 22 cancer types, representing:
- **Carcinomas** (epithelial origin): Lung, breast, colon, pancreas, etc.
- **Sarcomas** (mesenchymal origin): Bone, soft tissue
- **Leukemias/Lymphomas** (hematological): Blood cancers
- **Gliomas** (neural origin): Brain tumors
- **Melanomas** (melanocyte origin): Skin cancers

### DNA Methylation in Pan-Cancer Studies

Cross-cancer methylation analysis reveals:
- **Tissue-Specific Patterns**: Methylation reflects cell-of-origin
- **Universal Cancer Methylation**: CpG island methylator phenotype (CIMP)
- **Subtype Distinctions**: Epigenetic subtypes within histologies
- **Drug Response**: Methylation of DNA repair genes → chemosensitivity

### Clinical Relevance (with GDSC Data)

When integrated with drug response:
- **Precision Medicine**: Predict which drugs will work for specific methylation profiles
- **Resistance Mechanisms**: Identify epigenetic causes of drug resistance
- **Biomarker Development**: Methylation-based companion diagnostics
- **Drug Repurposing**: Find new uses for existing drugs based on methylation

---

## Integration with GDSC Database

### GDSC Data Structure

**Access**: www.cancerrxgene.org

**Files**:
- **Drug Response**: IC50, AUC values for 453 compounds × ~1,000 cell lines
- **Cell Line Annotations**: COSMIC IDs, tissue types, genomic features
- **Genomic Data**: Mutations, copy number, gene expression

### Linkage to GSE68379

**Key**: `cosmic_id` column in metadata

**Mapping**:
```python
# Example integration
import pandas as pd

# Load GDSC drug response
gdsc_response = pd.read_csv('GDSC_drug_response.csv')

# Load GSE68379 metadata
gse_metadata = pd.read_csv('data/processed/GSE68379_sample_metadata.csv')

# Merge on COSMIC ID
merged = pd.merge(
    gse_metadata,
    gdsc_response,
    left_on='cosmic_id',
    right_on='COSMIC_ID',
    how='inner'
)
```

**Expected Overlap**: ~700-900 cell lines (verify after GDSC download)

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

1. Yang W, et al. (2013). "Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells." *Nucleic Acids Research*, 41(Database issue):D955-961. DOI: 10.1093/nar/gks1111

2. Garnett MJ, et al. (2012). "Systematic identification of genomic markers of drug sensitivity in cancer cells." *Nature*, 483(7391):570-575. DOI: 10.1038/nature11005

3. GEO Accession: GSE68379 - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68379

4. GDSC Database: www.cancerrxgene.org

5. Project References: `.claude/references/cancer_cell_line_paper.pdf`

---

**Notes**:
- All file paths relative to project root: `bioml_working_repo/`
- Data files gitignored (not in version control)
- Processed data can be regenerated from raw data
- Random seed = 42 for all analyses (reproducibility)
- GDSC integration requires separate data download (not automated)
