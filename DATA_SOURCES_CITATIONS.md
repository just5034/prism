# Data Sources and Citations

**Central reference for all data sources used in this biomedical AI research project**

**Last Updated**: October 31, 2025
**Project**: Pharmacoepigenomics Biomarker Discovery

---

## Primary Datasets

### 1. GSE270494 - Hematological Malignancy Cell Lines DNA Methylation

**Description**: DNA methylation profiles of 180 human hematological malignancy cell lines using Illumina HumanMethylation450K array

**Citation**:
```
Noguera-Castells A, et al. (2025). A DNA methylation database of human and mouse
hematological malignancy cell lines. Leukemia.
PMID: [TBD - 2025 publication]
```

**Source**:
- **Database**: NCBI Gene Expression Omnibus (GEO)
- **Accession**: [GSE270494](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270494)
- **Platform**: GPL13534 (Illumina HumanMethylation450 BeadChip)
- **Samples**: 180 human cell lines
- **Features**: ~760,000 CpG sites

**Download Information**:
- **Files Used**: `GSE270494_Noguera-Castells_Average_Beta_Homo_Sapiens.csv.gz` (1.1 GB)
- **Downloaded From**: ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE270nnn/GSE270494/suppl/
- **Date Downloaded**: October 20, 2025
- **Local Path**: `data/raw/GSE270494/`

**License**: Public domain (GEO data)

**Usage in Project**:
- Cross-dataset validation
- Hematological malignancy-specific analysis
- Model testing on independent dataset

---

### 2. GSE68379 - Pan-Cancer Cell Lines DNA Methylation

**Description**: DNA methylation profiles of 1,028 cancer cell lines across 22 cancer types using Illumina HumanMethylation450K array

**Citation**:
```
Iorio F, Knijnenburg TA, et al. (2016). A Landscape of Pharmacogenomic Interactions
in Cancer. Cell, 166(3), 740-754.
DOI: 10.1016/j.cell.2016.06.017
PMID: 27397505
```

**Source**:
- **Database**: NCBI Gene Expression Omnibus (GEO)
- **Accession**: [GSE68379](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68379)
- **Platform**: GPL13534 (Illumina HumanMethylation450 BeadChip)
- **Samples**: 1,028 cancer cell lines
- **Features**: ~485,000 CpG sites

**Download Information**:
- **Files Used**: `GSE68379_Matrix.processed.txt.gz` (3.9 GB)
- **Downloaded From**: ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68379/suppl/
- **Date Downloaded**: October 20, 2025
- **Local Path**: `data/raw/GSE68379/`

**License**: Public domain (GEO data)

**Usage in Project**:
- Primary dataset for model training
- ML-ready dataset creation (10,000 top variable CpG sites)
- Multi-omics integration

---

### 3. GDSC - Genomics of Drug Sensitivity in Cancer (Gene Expression)

**Description**: RMA-normalized gene expression data for cancer cell lines from GDSC project using Affymetrix U219 microarray

**Citation**:
```
Yang W, Soares J, Greninger P, et al. (2013). Genomics of Drug Sensitivity in Cancer
(GDSC): a resource for therapeutic biomarker discovery in cancer cells.
Nucleic Acids Research, 41(Database issue), D955-D961.
DOI: 10.1093/nar/gks1111
PMID: 23180760
```

**Updated Release Citation**:
```
Garnett MJ, Edelman EJ, Heidorn SJ, et al. (2012). Systematic identification of
genomic markers of drug sensitivity in cancer cells. Nature, 483(7391), 570-575.
DOI: 10.1038/nature11005
PMID: 22460902
```

**Source**:
- **Website**: https://www.cancerrxgene.org/
- **Download Page**: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
- **Platform**: Affymetrix Human Genome U219 Array
- **Samples**: ~1,000 cancer cell lines
- **Features**: ~17,000 genes (RMA-processed)

**Download Information**:
- **Files Used**: `Cell_line_RMA_proc_basalExp.txt.zip` (137 MB compressed → 293 MB uncompressed)
- **Downloaded From**: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip
- **Date Downloaded**: October 31, 2025
- **Local Path**: `data/raw/CCLE/Cell_line_RMA_proc_basalExp.txt`

**Processing Notes**:
- Data is already RMA-normalized (log2 scale)
- Includes only protein-coding genes
- Cell lines indexed by COSMIC IDs (format: DATA.XXXXXX)

**License**: Open data (GDSC Terms of Use: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html)

**Usage in Project**:
- Multi-omics integration (methylation + expression)
- Feature enrichment for drug response prediction
- Cross-validation of methylation-based models

---

### 4. GDSC - Drug Response Data (IC50 Values)

**Description**: Fitted dose-response curves and IC50 values for hundreds of drugs tested across cancer cell lines

**Citation**:
```
Same as GDSC Gene Expression (Yang et al. 2013, Garnett et al. 2012)
```

**Source**:
- **Website**: https://www.cancerrxgene.org/
- **Download Page**: https://www.cancerrxgene.org/downloads/bulk_download
- **Drugs**: 450+ compounds
- **Cell Lines**: ~1,000 cancer cell lines

**Download Information**:
- **Files Used**: GDSC1 or GDSC2 fitted dose response files
- **Downloaded From**: Manual download from bulk download page
- **Status**: Integrated in Session 6 (see notebook 04_Integrate_GDSC_Drug_Response.ipynb)
- **Local Path**: `data/raw/GDSC/`

**Data Fields**:
- `LN_IC50`: Natural log of IC50 (primary metric)
- `AUC`: Area under dose-response curve
- `RMSE`: Root mean squared error of fit

**License**: Open data (GDSC Terms of Use)

**Usage in Project**:
- Target variable for drug response prediction models
- Pharmacoepigenomic biomarker discovery
- Drug sensitivity classification

---

## Secondary/Reference Databases

### 5. COSMIC - Catalogue of Somatic Mutations in Cancer

**Description**: Cell line annotations, COSMIC IDs, and metadata

**Citation**:
```
Tate JG, Bamford S, Jubb HC, et al. (2019). COSMIC: the Catalogue Of Somatic
Mutations In Cancer. Nucleic Acids Research, 47(D1), D941-D947.
DOI: 10.1093/nar/gky1015
PMID: 30371878
```

**Source**:
- **Website**: https://cancer.sanger.ac.uk/cosmic
- **Cell Line Project**: https://cancer.sanger.ac.uk/cell_lines

**Usage in Project**:
- Cell line identifier mapping (COSMIC IDs)
- Cross-referencing between datasets
- Metadata validation

**License**: Academic use (COSMIC license required for commercial use)

---

### 6. DepMap - Cancer Dependency Map (Future Use)

**Description**: CCLE expression data and multi-omics profiles

**Citation**:
```
DepMap, Broad (2025). DepMap 24Q4 Public. figshare. Dataset.
DOI: [Latest release]
```

**Source**:
- **Website**: https://depmap.org/
- **Portal**: https://depmap.org/portal/download/
- **FigShare**: https://figshare.com/authors/Broad_DepMap/5514062

**Download Information**:
- **Attempted**: DepMap 25Q2, 24Q4 releases
- **Status**: Download unsuccessful via automated methods
- **Alternative**: Used GDSC expression data instead

**Note**: DepMap and CCLE data overlap significantly with GDSC. We prioritized GDSC for consistency with drug response data.

**License**: CC BY 4.0

---

## Data Processing and Integration

### ML-Ready Datasets Created

#### Dataset 1: Methylation-Only
- **File**: `data/processed/ML_dataset_methylation_features.csv.gz`
- **Size**: 77.7 MB
- **Samples**: 1,028 cell lines
- **Features**: 10,000 CpG sites (top variable from GSE68379)
- **Metadata**: Primary site, histology, COSMIC ID
- **Created**: Session 6 (October 24, 2025)

####Dataset 2: Methylation + Drug Response
- **File**: `data/processed/ML_dataset_methylation_drug_response.csv.gz`
- **Samples**: Matched cell lines from GSE68379 and GDSC
- **Features**: 10,000 CpG sites + drug IC50 targets
- **Created**: Session 6 (see notebook 04)

#### Dataset 3: Multi-Omics (Methylation + Expression)
- **File**: `data/processed/ML_dataset_multiomics.csv.gz`
- **Status**: In progress (Session 7, October 31, 2025)
- **Planned Features**:
  - 10,000 CpG sites (methylation)
  - 5,000 genes (expression, top variable)
  - Metadata
- **Expected Samples**: ~900-1,000 (depending on overlap)

---

## Software and Tools

### Data Analysis
- **Python**: 3.13.1
- **pandas**: 2.2.3
- **numpy**: 2.2.2
- **scikit-learn**: 1.6.1
- **matplotlib**: 3.10.0
- **seaborn**: 0.13.2

### Data Acquisition
- **GEOparse**: 2.0.4 (for GEO data download)
- **curl**: Command-line HTTP/FTP downloads

### Development Environment
- **OS**: Windows (MINGW64_NT-10.0-26100)
- **IDE**: VS Code / Jupyter Notebook
- **Version Control**: Git

---

## Data Use Compliance

### Ethical Considerations
- All datasets are publicly available and deidentified
- Cell line data (not patient data)
- No human subjects research
- Compliant with NIH/NCBI data use policies

### Attribution Requirements

When publishing results using this data, include:

1. **For GSE270494**:
   - Cite Noguera-Castells et al. (2025)
   - Acknowledge GEO accession GSE270494

2. **For GSE68379**:
   - Cite Iorio et al. (2016), Cell
   - Acknowledge GEO accession GSE68379

3. **For GDSC Data**:
   - Cite Yang et al. (2013) and Garnett et al. (2012)
   - Acknowledge GDSC consortium
   - Include URL: https://www.cancerrxgene.org/

4. **For COSMIC**:
   - Cite Tate et al. (2019)
   - Acknowledge COSMIC database

### Example Acknowledgment Text

```
DNA methylation data were obtained from GSE270494 (Noguera-Castells et al., 2025)
and GSE68379 (Iorio et al., 2016) via the Gene Expression Omnibus. Gene expression
and drug response data were obtained from the Genomics of Drug Sensitivity in Cancer
(GDSC) database (Yang et al., 2013; Garnett et al., 2012). Cell line annotations
were obtained from the COSMIC database (Tate et al., 2019).
```

---

## Data Availability Statement (for Publication)

```
All data used in this study are publicly available. DNA methylation data are
available from the Gene Expression Omnibus (GEO) under accessions GSE270494 and
GSE68379 (https://www.ncbi.nlm.nih.gov/geo/). Gene expression and drug response
data are available from the Genomics of Drug Sensitivity in Cancer (GDSC) database
(https://www.cancerrxgene.org/). Processed ML-ready datasets created for this study
are available at [GitHub repository URL] under [license].
```

---

## Version Control

| Date | Session | Changes | Updated By |
|------|---------|---------|------------|
| 2025-10-31 | 7 | Initial creation, added GSE270494, GSE68379, GDSC expression | Claude |
| 2025-10-31 | 7 | Added GDSC drug response, COSMIC references | Claude |

---

## Contact for Data Questions

For questions about:
- **GSE270494**: Contact authors of Noguera-Castells et al. (2025)
- **GSE68379**: GEO curators (geo@ncbi.nlm.nih.gov)
- **GDSC**: gdsc@sanger.ac.uk
- **COSMIC**: cosmic@sanger.ac.uk

---

## Notes for Future Work

### Potential Additional Data Sources

1. **CCLE (Cancer Cell Line Encyclopedia)**
   - URL: https://sites.broadinstitute.org/ccle/
   - Overlap with GDSC but different drug panel
   - Contains protein expression (RPPA) data

2. **cBioPortal**
   - URL: https://www.cbioportal.org/
   - Mutation data for cell lines
   - Copy number variations

3. **CellMinerCDB**
   - URL: https://discover.nci.nih.gov/cellminercdb/
   - NCI-60 cell line data
   - Smaller but well-characterized panel

4. **PharmacoGx (R package)**
   - Harmonized multi-source pharmacogenomic data
   - Standardized access to GDSC, CCLE, gCSI, CTRPv2

### Data Integration Challenges

1. **Cell Line Name Mapping**:
   - Different databases use different naming conventions
   - COSMIC IDs provide consistent identifiers but may differ slightly
   - Manual curation may be required for some mappings

2. **Platform Differences**:
   - GSE270494 vs GSE68379: Different CpG coverage (450K vs 450K)
   - Expression: Microarray (GDSC) vs RNA-seq (DepMap)

3. **Missing Data**:
   - Not all cell lines have all data types
   - Drug testing coverage varies by drug
   - Requires careful handling in ML models

---

**Document Maintained By**: Biomedical AI Research Team
**Last Review**: October 31, 2025
**Next Review**: Before manuscript submission
