# ML Dataset Characteristics Reference

**Dataset**: `data/processed/ml_with_gene_expr.csv`
**Created**: November 2025
**Purpose**: Multi-omics cancer cell line dataset for machine learning

---

## Dataset Structure

### Dimensions
- **Rows**: 987 cancer cell lines
- **Columns**: 15,003 total
  - 3 metadata columns
  - 10,000 methylation features (CpG sites)
  - 5,000 gene expression features

### Row Index
- **Type**: String (cell line name)
- **Examples**: '697', '5637', '201T', '22RV1', 'A549', 'HeLa'
- **Meaning**: Unique identifier for each cancer cell line

### Column Types

#### 1. Metadata Columns (3)
| Column Name | Data Type | Description |
|------------|-----------|-------------|
| `primary site` | object (string) | Primary anatomical site of cancer origin |
| `primary histology` | object (string) | Specific histological cancer type |
| `cosmic_id` | float64 | COSMIC database identifier for the cell line |

#### 2. Methylation Features (10,000)
- **Column Pattern**: `cg########` (e.g., `cg00944421`, `cg14557185`)
- **Data Type**: float64
- **Value Range**: 0.0 to 1.0 (beta values)
- **Meaning**: DNA methylation level at specific CpG site
  - 0.0 = unmethylated
  - 1.0 = fully methylated
  - Intermediate values = partial methylation
- **Source**: Illumina HumanMethylation450K array (GSE68379)
- **Selection**: Top 10,000 most variable CpG sites across all cell lines

#### 3. Gene Expression Features (5,000)
- **Column Pattern**: `expr_<GENE_SYMBOL>` (e.g., `expr_TP53`, `expr_EGFR`)
- **Data Type**: float64
- **Value Range**: Continuous (RMA-normalized, log2 scale)
- **Meaning**: Gene expression level (higher = more expressed)
- **Source**: Affymetrix U219 microarray, RMA-processed (GDSC)
- **Selection**: Top 5,000 most variable genes across all cell lines

---

## Available Characteristics for Stratification

### A. Current Dataset Characteristics (in `ml_with_gene_expr.csv`)

#### 1. Primary Site
- **Column**: `primary site`
- **Data Type**: Categorical
- **Coverage**: 987/987 samples (100%)
- **Unique Categories**: 13

**Distribution**:
| Category | Count | Percentage |
|----------|-------|------------|
| lung | 192 | 19.5% |
| blood | 171 | 17.3% |
| urogenital_system | 112 | 11.3% |
| digestive_system | 101 | 10.2% |
| nervous_system | 89 | 9.0% |
| aero_digestive_tract | 80 | 8.1% |
| skin | 55 | 5.6% |
| breast | 50 | 5.1% |
| bone | 36 | 3.6% |
| kidney | 33 | 3.3% |
| pancreas | 31 | 3.1% |
| soft_tissue | 21 | 2.1% |
| thyroid | 16 | 1.6% |

#### 2. Primary Histology
- **Column**: `primary histology`
- **Data Type**: Categorical
- **Coverage**: 987/987 samples (100%)
- **Unique Categories**: 56

**Distribution (Top 20)**:
| Category | Count |
|----------|-------|
| lung_NSCLC_adenocarcinoma | 66 |
| lung_small_cell_carcinoma | 59 |
| glioma | 53 |
| melanoma | 52 |
| breast | 50 |
| large_intestine | 49 |
| ovary | 45 |
| head and neck | 44 |
| oesophagus | 36 |
| B_cell_lymphoma | 32 |
| kidney | 32 |
| pancreas | 31 |
| neuroblastoma | 31 |
| stomach | 28 |
| acute_myeloid_leukaemia | 26 |
| ewings_sarcoma | 21 |
| mesothelioma | 21 |
| bladder | 20 |
| liver | 17 |
| cervix | 16 |

**All 56 Categories**: lymphoblastic_leukemia, bladder, lung_NSCLC_adenocarcinoma, prostate, stomach, glioma, melanoma, head and neck, kidney, thyroid, rhabdomyosarcoma, lung_NSCLC, leukemia, myeloma, B_cell_lymphoma, lung_small_cell_carcinoma, neuroblastoma, large_intestine, breast, mesothelioma, ovary, oesophagus, osteosarcoma, carcinoid, T_cell_lymphoma, cervix, liver, lymphoma, lung_NSCLC_squamous_cell_carcinoma, acute_lymphoblastic_leukemia, acute_myeloid_leukaemia, leukemia_other, pancreas, biliary_tract, chronic_myelogenous_leukemia, meningioma, Hodgkin_lymphoma, primitive_neuroectodermal_tumor, ewings_sarcoma, chronic_lymphocytic_leukemia, rhabdoid_tumor, chondrosarcoma, hairy_cell_leukemia, neuroepithelioma, medulloblastoma, thymic, lymphoma_Burkitt, lymphoblastic_T_cell_acute_lymphoblastic_leukemia, chordoma, lung_adenosquamous, acute_lymphoblastic_B_cell_leukemia, lymphoblastic_B_cell_acute_lymphoblastic_leukemia, fibroblast, plasma_cell_myeloma, malignant_melanoma, non_small_cell_lung_cancer

#### 3. COSMIC ID
- **Column**: `cosmic_id`
- **Data Type**: Numeric (float64)
- **Coverage**: 987/987 samples (100%)
- **Unique Values**: 987 (unique per cell line)
- **Meaning**: Database identifier linking to COSMIC cell line database
- **Usage**: For matching with external databases (GDSC, COSMIC, DepMap)

---

### B. GDSC Annotation Characteristics (via COSMIC ID matching)

**Source**: `data/raw/CCLE/GDSC_cell_lines_annotations.xlsx`
**Matching**: Based on COSMIC ID
**Coverage**: 950/987 samples (96.3%) have GDSC annotations

#### 4. Tissue Descriptor Level 1
- **Source Column**: GDSC "Tissue descriptor 1"
- **Data Type**: Categorical
- **Coverage**: 950/987 samples (96.3%)
- **Unique Categories**: 19

**Distribution**:
| Category | Count |
|----------|-------|
| lung_NSCLC | 111 |
| urogenital_system | 105 |
| leukemia | 85 |
| aero_dig_tract | 79 |
| lymphoma | 70 |
| lung_SCLC | 66 |
| skin | 58 |
| nervous_system | 57 |
| digestive_system | 52 |
| breast | 52 |
| large_intestine | 51 |
| bone | 40 |
| kidney | 34 |
| pancreas | 32 |
| neuroblastoma | 32 |
| (4 additional categories) | <30 each |

#### 5. Tissue Descriptor Level 2
- **Source Column**: GDSC "Tissue descriptor 2"
- **Data Type**: Categorical
- **Coverage**: 950/987 samples (96.3%)
- **Unique Categories**: 55

**Distribution (Top 15)**:
| Category | Count |
|----------|-------|
| lung_NSCLC_adenocarcinoma | 67 |
| lung_small_cell_carcinoma | 66 |
| melanoma | 55 |
| glioma | 53 |
| breast | 52 |
| large_intestine | 51 |
| head and neck | 44 |
| ovary | 43 |
| oesophagus | 35 |
| B_cell_lymphoma | 35 |
| kidney | 33 |
| pancreas | 32 |
| neuroblastoma | 32 |
| stomach | 29 |
| acute_myeloid_leukaemia | 27 |

#### 6. TCGA Cancer Type Label
- **Source Column**: GDSC "Cancer Type (matching TCGA label)"
- **Data Type**: Categorical
- **Coverage**: 782/987 samples (79.2%)
- **Unique Categories**: 31

**Distribution (Top 15)**:
| Category | Count |
|----------|-------|
| SCLC | 66 |
| LUAD | 64 |
| SKCM | 55 |
| BRCA | 51 |
| COAD/READ | 51 |
| HNSC | 42 |
| GBM | 36 |
| DLBC | 35 |
| ESCA | 35 |
| OV | 34 |
| NB | 32 |
| KIRC | 32 |
| PAAD | 30 |
| LAML | 28 |
| ALL | 26 |

**All TCGA Labels**: SCLC, LUAD, SKCM, BRCA, COAD/READ, HNSC, GBM, DLBC, ESCA, OV, NB, KIRC, PAAD, LAML, ALL, UCEC, SARC, LIHC, CLL, BLCA, MM, THCA, STAD, MESO, LCML, AML, PRAD, LGG, CESC, UCS, ACC

#### 7. Microsatellite Instability (MSI) Status
- **Source Column**: GDSC "Microsatellite instability Status"
- **Data Type**: Categorical (binary)
- **Coverage**: 948/987 samples (96.1%)
- **Unique Categories**: 2

**Distribution**:
| Category | Count | Percentage |
|----------|-------|------------|
| MSS/MSI-L (stable) | 926 | 97.7% |
| MSI-H (high instability) | 60 | 6.3% |

**Meaning**:
- MSS/MSI-L: Microsatellite stable or low instability (genomically stable)
- MSI-H: Microsatellite instability-high (genomically unstable, hypermutated)

#### 8. Growth Properties
- **Source Column**: GDSC "Growth Properties"
- **Data Type**: Categorical
- **Coverage**: 950/987 samples (96.3%)
- **Unique Categories**: 3

**Distribution**:
| Category | Count | Percentage |
|----------|-------|------------|
| Adherent | 725 | 76.3% |
| Suspension | 244 | 25.7% |
| Semi-Adherent | 30 | 3.2% |

**Meaning**:
- Adherent: Cells that grow attached to culture surface
- Suspension: Cells that grow floating in culture medium
- Semi-Adherent: Cells with intermediate attachment properties

#### 9. Screen Medium
- **Source Column**: GDSC "Screen Medium"
- **Data Type**: Categorical
- **Coverage**: 950/987 samples (96.3%)
- **Unique Categories**: 5

**Distribution**:
| Category | Count |
|----------|-------|
| R | 526 |
| D/F12 | 429 |
| R+10 | 30 |
| D/F12+10 | 8 |
| H | 6 |

**Meaning**: Culture medium used for drug screening
- R: RPMI medium
- D/F12: DMEM/F12 medium
- R+10: RPMI with 10% serum
- D/F12+10: DMEM/F12 with 10% serum
- H: Other/specialized medium

---

### C. GDSC Drug Response Characteristics

**Source**: `data/raw/GDSC1_fitted_dose_response_27Oct23.xlsx`
**Note**: These characteristics are drug-specific, not cell-line-specific

#### 10. Drug Name
- **Source Column**: `DRUG_NAME`
- **Data Type**: Categorical
- **Unique Values**: 378 drugs

#### 11. Putative Drug Target
- **Source Column**: `PUTATIVE_TARGET`
- **Data Type**: Categorical
- **Unique Values**: 288 distinct targets

**Top 10 Targets**:
| Target | Drug-Cell Line Pairs |
|--------|---------------------|
| MEK1, MEK2 | 6,276 |
| BRAF | 4,873 |
| PARP1, PARP2 | 4,585 |
| HSP90 | 4,536 |
| AKT1, AKT2, AKT3 | 3,932 |
| MET | 3,159 |
| EGFR | 3,095 |
| HDAC1 | 3,056 |
| IGF1R, IR | 3,040 |
| ATM | 2,771 |

#### 12. Drug Pathway
- **Source Column**: `PATHWAY_NAME`
- **Data Type**: Categorical
- **Unique Values**: 24 pathways

**All Pathways**:
| Pathway | Drug-Cell Line Pairs |
|---------|---------------------|
| Other | 41,791 |
| Other, kinases | 39,374 |
| RTK signaling | 35,556 |
| PI3K/MTOR signaling | 32,543 |
| DNA replication | 17,194 |
| Chromatin histone acetylation | 16,773 |
| Cell cycle | 16,472 |
| Mitosis | 15,916 |
| Genome integrity | 15,547 |
| ERK MAPK signaling | 15,294 |
| Apoptosis regulation | 10,678 |
| Cytoskeleton | 8,537 |
| Unclassified | 8,246 |
| WNT signaling | 8,082 |
| Metabolism | 7,112 |
| JNK and p38 signaling | 6,851 |
| Chromatin histone methylation | 6,331 |
| Protein stability and degradation | 6,260 |
| Chromatin other | 6,231 |
| EGFR signaling | 5,841 |
| Hormone-related | 4,479 |
| p53 pathway | 3,638 |
| IGF1R signaling | 3,087 |
| ABL signaling | 1,328 |

#### 13. TCGA Label (from drug response data)
- **Source Column**: `TCGA_DESC` (in drug response file)
- **Data Type**: Categorical
- **Unique Values**: 31 cancer types

**Top 10**:
| TCGA Label | Drug-Cell Line Pairs |
|------------|---------------------|
| UNCLASSIFIED | 63,255 |
| LUAD | 21,547 |
| SCLC | 19,808 |
| SKCM | 18,975 |
| BRCA | 17,288 |
| COREAD | 15,529 |
| HNSC | 13,429 |
| ESCA | 12,136 |
| GBM | 11,945 |
| DLBC | 11,430 |

---

## Data Sources

### Primary Data
1. **GSE68379** - DNA methylation (1,028 cancer cell lines)
   - Platform: Illumina HumanMethylation450K
   - Features: ~485,000 CpG sites (10,000 most variable selected)

2. **GDSC Gene Expression** - RMA-normalized microarray data
   - Platform: Affymetrix U219
   - Features: ~17,000 genes (5,000 most variable selected)

### Annotation Data
3. **GDSC Cell Line Annotations** - Cell line metadata (1,002 cell lines)
   - Tissue descriptors, TCGA labels, MSI status, growth properties, culture conditions

4. **GDSC Drug Response** - IC50 values for 378 drugs
   - Drug targets, pathways, pharmacological annotations

### Matching
- Cell lines matched across datasets using **COSMIC ID**
- Final integrated dataset: 987 cell lines with both methylation and expression data
- 950/987 (96.3%) have complete GDSC annotations

---

## Notes

- All categorical characteristics are available for stratified train-test splitting
- Coverage indicates how many of the 987 cell lines have values for each characteristic
- Cell line names (row indices) are standardized to match GDSC nomenclature
- COSMIC IDs enable linking to additional external databases (COSMIC, DepMap, CCLE)


