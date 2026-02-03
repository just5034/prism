# Pharmacoepigenomics EDA Findings & AI/ML Research Directions

**Date**: October 20, 2025
**Analyst**: Computational Biology & ML Engineering Team
**Datasets**: GSE270494 (Hematological Malignancy) & GSE68379 (Cancer Cell Lines)

---

## Executive Summary

We have completed exploratory data analysis (EDA) on two complementary DNA methylation datasets in cancer cell lines. The analyses reveal significant opportunities for AI/ML-driven biomarker discovery, drug response prediction, and novel methodological development. This document synthesizes key findings and proposes concrete research directions that could yield publishable results and clinical impact.

**Key Opportunities Identified:**
- Cross-dataset validation framework for pharmacoepigenetic biomarkers
- Multi-omics integration potential with existing drug response databases
- Deep learning architectures for methylation pattern recognition
- Novel feature engineering for CpG site selection
- Transfer learning across cancer types

---

## Part 1: Dataset Summaries & Key Findings

### 1.1 GSE270494: Hematological Malignancy Cell Lines

**Dataset Characteristics:**
- **Samples**: 180 human cell lines (71 annotated with disease types)
- **Coverage**: 760,090 CpG sites (Infinium HumanMethylation450K/Mouse array)
- **Disease Types**: 10 categories (AML, DLBCL, B-ALL, T-ALL, MM, HL, MCL, BL, CML, TCL)
- **Data Quality**: Excellent (0% missing values, all beta-values in [0,1] range)

**Methylation Characteristics:**
- **Mean methylation**: 0.592 (bimodal distribution typical for DNA methylation)
- **Highly variable CpG sites**: 95,384 sites with variance > 0.1 (potential biomarkers)
- **Distribution**: 52.5% hypermethylated, 31.2% hypomethylated, 16.3% intermediate

**Dimensionality Reduction Findings:**
- **PC1 + PC2 variance**: 45.1% (strong primary signal)
- **Disease separation**: Visible clustering by disease type in PCA space
- **Biological validation**: Hierarchical clustering confirms disease-specific methylation patterns

**Key Observations:**
1. ✅ Strong methylation-based disease signatures exist
2. ✅ High variance sites are enriched (likely in regulatory regions based on literature)
3. ⚠️ 109/180 samples lack disease annotations (opportunity for semi-supervised learning)
4. 🔬 Reference paper identified 802 drug-associated methylation regions (dDMRs)

---

### 1.2 GSE68379: Pan-Cancer Cell Line Pharmacoepigenomics

**Dataset Characteristics:**
- **Samples**: 1,028 cancer cell lines (100% annotated)
- **Coverage**: 485,512 CpG sites (Infinium HumanMethylation450K)
- **Cancer Types**: 13 primary sites, 56 distinct histologies
- **Data Quality**: Very good (0.0095% missing values, handled via imputation)

**Methylation Characteristics:**
- **Mean methylation**: 0.506 (slightly lower than hematological dataset)
- **Distribution**: 34.4% hypermethylated, 30.0% hypomethylated, 35.6% intermediate
- **Variance**: 10,000 most variable CpGs identified for downstream analysis

**Dimensionality Reduction Findings:**
- **PC1 + PC2 variance**: 31.6% (more heterogeneous than GSE270494)
- **Primary site clustering**: Clear separation of some tissue types (e.g., blood, lung, nervous system)
- **Histology patterns**: Hierarchical clustering reveals histology-specific methylation signatures

**Key Observations:**
1. ✅ Pan-cancer methylation atlas with rich clinical annotations
2. ✅ Integration with GDSC drug response data available (453 drugs tested)
3. 🔬 Higher sample heterogeneity offers opportunities for multi-task learning
4. 📊 Large sample size (n=1,028) enables deep learning approaches

---

### 1.3 Comparative Insights

| Characteristic | GSE270494 (Hematological) | GSE68379 (Pan-Cancer) |
|---------------|---------------------------|----------------------|
| **Sample size** | 180 | 1,028 |
| **CpG coverage** | 760,090 | 485,512 |
| **Overlap** | ~485K CpG sites common | ~485K CpG sites common |
| **Mean methylation** | 0.592 | 0.506 |
| **PC1+PC2 variance** | 45.1% | 31.6% |
| **Disease focus** | Hematological malignancies | Pan-cancer (22 types) |
| **Drug data** | 186 drugs (from paper) | 453 drugs (GDSC linked) |

**Critical Observation**:
- **Hematological overlap**: GSE68379 contains 177 blood cancer cell lines vs. 180 in GSE270494
- **Opportunity**: Direct comparison of methylation patterns in overlapping cell lines
- **Validation strategy**: Use GSE270494 as discovery, GSE68379 blood samples as validation

---

## Part 2: Proposed AI/ML Research Directions

### 2.1 **HIGH PRIORITY: Cross-Dataset Pharmacoepigenetic Biomarker Discovery**

**Objective**: Identify robust DNA methylation biomarkers predictive of drug response that generalize across cancer types.

**Approach**:
```
1. Differential Methylation Analysis
   - Identify CpG sites associated with drug sensitivity in GSE68379
   - Validate using drug response data from GSE270494 literature
   - Apply FDR correction (q < 0.05) for multiple testing

2. Feature Selection Pipeline
   - Variance filtering (top 10K variable CpGs)
   - LASSO regression for sparse feature selection
   - Recursive feature elimination (RFE) for optimal biomarker sets
   - Compare with reference paper's 802 dDMRs

3. Predictive Modeling
   - Random Forest classifier for drug sensitivity (baseline)
   - Gradient Boosting (XGBoost/LightGBM) for better performance
   - Elastic Net for interpretable linear models
   - Evaluate using stratified cross-validation (10-fold)

4. Cross-Dataset Validation
   - Train on GSE68379, validate on GSE270494 blood cancers
   - Assess generalizability of methylation-drug associations
   - Compute confidence intervals for all performance metrics
```

**Expected Outcomes**:
- Ranked list of CpG biomarkers for specific drug classes
- Validation that methylation predicts response across datasets
- Publishable results comparing to reference papers

**Computational Requirements**: Moderate (can run on local machine with 16GB RAM)

---

### 2.2 **INNOVATIVE: Deep Learning for Methylation Pattern Recognition**

**Objective**: Develop convolutional neural network (CNN) or transformer-based models to learn spatial methylation patterns that traditional feature selection misses.

**Biological Rationale**:
- CpG sites are not independent; regional methylation matters (promoters, enhancers, CpG islands)
- Genomic context contains information (sequence, chromatin state, transcription factor binding)
- Deep learning can capture complex, non-linear interactions

**Proposed Architectures**:

**A. 1D-CNN for Regional Methylation Patterns**
```python
# Concept: Treat CpG sites as sequence data
# Input: [n_samples, n_cpgs_in_region, 1]
# (e.g., 2kb windows around gene promoters)

Architecture:
- Conv1D layers (kernel_size=3, 5, 7) to capture local correlations
- MaxPooling to identify most informative CpG clusters
- Dense layers for drug response classification
- Multi-task learning: predict multiple drug sensitivities simultaneously
```

**B. Graph Neural Network (GNN) for CpG Interaction Networks**
```
# Concept: CpGs as nodes, edges based on:
# - Genomic distance (< 1kb)
# - Co-methylation patterns (correlation > 0.7)
# - Shared regulatory elements (from ENCODE/Roadmap)

Architecture:
- Graph Convolutional Network (GCN) or Graph Attention Network (GAT)
- Node features: methylation values + genomic annotations
- Edge weights: biological relationships
- Output: Drug response prediction or cancer type classification
```

**C. Transformer for Long-Range Dependencies**
```
# Hypothesis: Distant CpG sites interact via chromatin looping
# Transformers excel at capturing long-range dependencies

Architecture:
- Self-attention mechanism over top variable CpG sites
- Positional encoding based on genomic coordinates
- Pre-training on large methylation datasets (TCGA, GEO)
- Fine-tuning on GSE68379 for drug response prediction
```

**Implementation Plan**:
1. **Start Simple**: 1D-CNN on promoter regions (±2kb of TSS) for top 1000 genes
2. **Incorporate Biology**: Add genomic annotations (CpG islands, chromatin states, TF binding sites)
3. **Scale Up**: GNN using Hi-C data for chromatin interaction networks
4. **Transfer Learning**: Pre-train on TCGA methylation data (>10,000 samples)

**Expected Novelty**:
- First GNN application to pharmacoepigenomics
- Biologically-informed architecture that outperforms traditional ML
- Potential Nature Communications/Bioinformatics publication

**Computational Requirements**: High (requires GPU, ~1-2 days training)

---

### 2.3 **TRANSLATIONAL: Drug Repurposing via Methylation Similarity**

**Objective**: Identify drug repurposing opportunities by finding methylation patterns shared across diseases.

**Approach**:
```
1. Methylation Signature Database
   - Extract disease-specific methylation signatures from GSE270494
   - Create drug response signatures from GSE68379
   - Store in queryable database (e.g., SQLite or MongoDB)

2. Similarity Search Algorithm
   - Compute pairwise methylation distances (Euclidean, cosine, Pearson)
   - Use dimensionality reduction (t-SNE, UMAP) for visualization
   - Identify "nearest neighbor" diseases or cell lines

3. Drug Repurposing Hypothesis Generation
   - Example: If AML and lung cancer share methylation patterns,
     test if lung cancer drugs (from GDSC) work on AML cells
   - Prioritize candidates using pathway enrichment analysis

4. Validation Strategy
   - Cross-reference with ClinicalTrials.gov for ongoing trials
   - Check literature for existing evidence
   - Propose in vitro validation experiments
```

**Example Use Case**:
```
Query: "Find drugs that target methylation patterns similar to DLBCL"
Output:
  1. DLBCL methylation signature (500 CpG sites)
  2. Most similar cell lines from GSE68379 (e.g., other B-cell cancers)
  3. Drugs effective in similar cell lines (from GDSC)
  4. Biological rationale (shared pathway enrichments)
```

**Expected Outcomes**:
- Web tool or API for methylation-based drug repurposing
- List of high-confidence repurposing candidates
- Collaboration opportunities with experimental labs

**Computational Requirements**: Low to Moderate

---

### 2.4 **METHODOLOGICAL: Transfer Learning Framework for Rare Cancers**

**Objective**: Develop transfer learning approach to predict drug response in rare hematological malignancies (small sample sizes) using knowledge from large pan-cancer dataset.

**Problem**:
- Some GSE270494 disease types have only 2-6 samples (TCL, CML, BL, MCL)
- Insufficient data for robust ML model training
- Traditional approaches would fail or overfit

**Solution**: Transfer Learning Pipeline
```
Phase 1: Pre-training on GSE68379 (n=1,028)
  - Train deep neural network to predict:
    a) Primary site classification (13 classes)
    b) Histology classification (56 classes)
    c) Drug response (multi-task, 453 drugs)
  - Extract learned methylation features (embedding layer)

Phase 2: Fine-tuning on GSE270494 rare diseases
  - Freeze early layers (general methylation patterns)
  - Re-train final layers on small disease cohorts
  - Use data augmentation (synthetic minority oversampling)

Phase 3: Evaluation
  - Leave-one-out cross-validation for rare diseases
  - Compare performance vs. training from scratch
  - Assess feature importance to understand what transfers
```

**Technical Implementation**:
```python
# Pseudo-code
# Step 1: Pre-train on large dataset
model_pretrain = Sequential([
    Dense(512, activation='relu', input_dim=10000),  # 10K variable CpGs
    Dropout(0.3),
    Dense(256, activation='relu', name='embedding_layer'),
    Dropout(0.3),
    Dense(n_classes, activation='softmax')  # 13 primary sites
])
model_pretrain.fit(X_gse68379, y_primary_site, epochs=50)

# Step 2: Extract embeddings
embedding_model = Model(
    inputs=model_pretrain.input,
    outputs=model_pretrain.get_layer('embedding_layer').output
)
X_gse270494_embedded = embedding_model.predict(X_gse270494)

# Step 3: Train classifier on embeddings (small data regime)
from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X_gse270494_embedded, y_disease_type)
```

**Expected Novelty**:
- First transfer learning application for rare cancer methylation analysis
- Demonstrates that pan-cancer knowledge improves rare disease predictions
- Generalizable framework for other rare disease -omics studies

**Publication Target**: Bioinformatics (Methods paper)

---

### 2.5 **EXPLORATORY: Multi-Omics Integration with Public Databases**

**Objective**: Integrate methylation data with gene expression, mutation, and copy number variation to build holistic predictive models.

**Data Sources to Integrate**:
```
1. Gene Expression (RNA-seq)
   - CCLE (Cancer Cell Line Encyclopedia) - same cell lines as GSE68379
   - GEO: GSE36133 (expression for hematological cell lines)

2. Mutation Data
   - COSMIC database (linked via COSMIC IDs in GSE68379 metadata)
   - DepMap mutation calls

3. Drug Response Data
   - GDSC (Genomics of Drug Sensitivity in Cancer) - 453 drugs
   - CTRP (Cancer Therapeutics Response Portal)

4. Pathway/Functional Annotations
   - MSigDB (Molecular Signatures Database)
   - KEGG, Reactome, Gene Ontology
```

**Integration Approaches**:

**A. Early Fusion (Concatenate Features)**
```
Input: [methylation (10K CpGs) + expression (5K genes) + mutations (500 genes)]
Model: Gradient Boosting or Deep Neural Network
Output: Drug response prediction
```

**B. Late Fusion (Ensemble of Modality-Specific Models)**
```
Model 1: Methylation-only Random Forest
Model 2: Expression-only Random Forest
Model 3: Mutation-only Logistic Regression
Final: Meta-learner combining predictions (stacking)
```

**C. Multi-View Learning (Advanced)**
```
Architecture: Separate neural networks per data type
- Methylation pathway → 128-dim embedding
- Expression pathway → 128-dim embedding
- Mutation pathway → 64-dim embedding
Fusion: Concatenate embeddings + joint prediction layer
Loss: Multi-task (drug response + cancer type + pathway activity)
```

**Example Analysis**:
```
Research Question: "Do methylation and expression provide complementary information?"

Experiment:
1. Train models using:
   - Methylation only
   - Expression only
   - Both (early fusion)

2. Evaluate on drug response prediction (GDSC)

3. Analyze feature importance:
   - Which CpG sites matter when expression is included?
   - Do they regulate the same genes?

4. Biological validation:
   - Check if methylation-expression correlations match known biology
   - Example: Hypermethylated promoters → downregulated genes
```

**Expected Outcomes**:
- Quantification of multi-omics synergy for drug response prediction
- Identification of genes regulated by both methylation and expression
- Improved predictive performance over single-omics models

**Computational Requirements**: Moderate to High (depending on integration method)

---

### 2.6 **TOOL DEVELOPMENT: Interactive Methylation Explorer Dashboard**

**Objective**: Build web-based tool for scientists to explore methylation patterns and generate hypotheses.

**Core Features**:
```
1. Data Exploration
   - Upload custom methylation data or select from GSE270494/GSE68379
   - Interactive PCA/UMAP visualization (colored by disease, drug response, etc.)
   - Differential methylation volcano plots
   - Heatmap with customizable clustering

2. Biomarker Discovery Module
   - Select disease/drug of interest
   - Run automated differential methylation analysis
   - Export ranked CpG list with genomic annotations
   - Link to UCSC Genome Browser for visualization

3. Prediction Module
   - Upload new cell line methylation profile
   - Predict: cancer type, drug sensitivity, closest reference sample
   - Confidence intervals and uncertainty quantification
   - Downloadable report (PDF)

4. Drug Repurposing Module
   - Query similar methylation patterns
   - Suggest drugs based on nearest neighbors
   - Display pathway enrichment for selected CpG sets
```

**Technology Stack**:
```
Frontend: React.js or Streamlit (rapid prototyping)
Backend: Flask/FastAPI (Python)
Database: PostgreSQL (sample metadata) + HDF5 (methylation matrices)
Visualization: Plotly, D3.js
Deployment: Docker + AWS EC2 or Heroku
```

**Implementation Timeline**:
```
Week 1-2: Data pipeline + API development
Week 3-4: Basic UI + PCA/clustering visualizations
Week 5-6: Predictive modeling integration
Week 7-8: Testing + documentation + deployment
```

**Impact**:
- Democratizes access to large-scale methylation data
- Accelerates hypothesis generation for wet-lab scientists
- Potential for high citation count (if published with NAR Web Server issue)

**Publication Target**: Nucleic Acids Research (Web Server Issue)

---

### 2.7 **BIOLOGICAL VALIDATION: Pathway Enrichment & Regulatory Network Analysis**

**Objective**: Annotate differentially methylated CpG sites with biological context to generate mechanistic hypotheses.

**Analysis Workflow**:
```
1. Map CpG Sites to Genes
   - Use Illumina 450K array annotation (from manufacturer)
   - Categories: TSS1500, TSS200, 5'UTR, 1stExon, Gene Body, 3'UTR
   - Relation to CpG Island: Island, N_Shore, S_Shore, N_Shelf, S_Shelf, OpenSea

2. Genomic Context Enrichment
   - Test if significant CpGs are enriched in:
     • CpG islands (regulatory regions)
     • Promoters (±2kb of TSS)
     • Enhancers (from ENCODE, Roadmap Epigenomics)
     • DNase I hypersensitive sites (open chromatin)
   - Compare to background distribution (Fisher's exact test)

3. Gene Set Enrichment Analysis (GSEA)
   - Input: Genes associated with differentially methylated CpGs
   - Test against:
     • KEGG pathways
     • Reactome pathways
     • Gene Ontology (Biological Process, Molecular Function)
     • MSigDB Hallmark gene sets
   - Multiple testing correction: FDR < 0.05

4. Transcription Factor Binding Site Analysis
   - Hypothesis: Differentially methylated regions block TF binding
   - Use HOMER or MEME suite to find enriched motifs
   - Cross-reference with ChIP-seq data (ENCODE)
   - Example: "CpG hypermethylation in E2F binding sites → cell cycle dysregulation"

5. Integration with Known Cancer Genes
   - Cross-reference with:
     • COSMIC Cancer Gene Census
     • OncoKB (MSK precision oncology database)
     • CancerMine (literature-mined cancer genes)
   - Question: Are our CpG biomarkers in known cancer genes?
```

**Example Biological Question**:
```
"Why do certain drugs only work in specific hematological malignancies?"

Approach:
1. Identify CpG sites associated with drug X sensitivity (from GSE68379)
2. Map to genes → find they cluster in "JAK-STAT signaling pathway"
3. Check if pathway is:
   - Enriched in responsive cell lines (GSEA)
   - Confirmed by expression data (if available)
   - Supported by literature (PubMed search)
4. Hypothesis: Drug X requires active JAK-STAT, which is methylation-regulated
5. Validation: Test prediction in GSE270494 cell lines
```

**Tools to Use**:
```python
# Python libraries
- mygene (gene annotation)
- gseapy (pathway enrichment)
- pyensembl (genomic annotations)
- biomart (Ensembl queries)

# R libraries (if needed)
- ChIPseeker (genomic annotation)
- clusterProfiler (enrichment analysis)
- methylGSA (methylation-specific GSEA)
```

**Expected Outcomes**:
- Biological interpretation of methylation biomarkers
- Pathway-level understanding of drug mechanisms
- Hypotheses for experimental validation
- Enhanced credibility for publication (reviewers love biological validation)

---

## Part 3: Prioritized Roadmap for Next Steps

### Immediate Next Steps (Week 1-2)

**Priority 1: Cross-Dataset Comparison**
```
Tasks:
□ Identify overlapping cell lines between GSE270494 and GSE68379
□ Align CpG site IDs (both use 450K array → should have ~485K overlap)
□ Compute correlation of methylation values for same cell lines
□ Investigate discrepancies (batch effects? technical variation?)

Deliverable:
- Jupyter notebook: "03_Cross_Dataset_Validation.ipynb"
- Venn diagram of sample overlap
- Correlation plots for matched samples
```

**Priority 2: Differential Methylation Analysis**
```
Tasks:
□ For GSE270494: Find CpG sites differentially methylated between disease types
  - Use limma or scipy.stats.ttest_ind
  - Apply multiple testing correction (FDR < 0.05)
  - Annotate significant CpGs with genomic features

□ For GSE68379: Identify CpG sites associated with drug response
  - Download GDSC drug response data
  - Correlate methylation with IC50 values
  - Focus on drugs mentioned in reference papers

Deliverable:
- CSV files: "GSE270494_differential_CpGs.csv", "GSE68379_drug_associated_CpGs.csv"
- Volcano plots and MA plots
- Heatmaps of top 100 differentially methylated CpGs
```

**Priority 3: Baseline Predictive Models**
```
Tasks:
□ Disease classification (GSE270494)
  - Input: Methylation beta-values (top 1000 variable CpGs)
  - Output: Disease type (10 classes)
  - Models: Random Forest, XGBoost, SVM
  - Evaluation: 10-fold stratified CV, confusion matrix, ROC curves

□ Cancer type classification (GSE68379)
  - Input: Methylation beta-values
  - Output: Primary site (13 classes) or Histology (56 classes)
  - Models: Same as above
  - Benchmark: Compare to PCA-based clustering

Deliverable:
- Jupyter notebook: "04_Baseline_Classification_Models.ipynb"
- Performance metrics table (accuracy, precision, recall, F1, AUC)
- Feature importance plots (which CpGs matter most?)
```

---

### Short-Term Goals (Weeks 3-6)

**Goal 1: Drug Response Prediction**
```
Objective: Predict drug sensitivity using methylation data

Data Integration:
1. Download GDSC drug response data (IC50 values)
   - API: https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
   - Match cell line names to GSE68379 samples

2. Select high-confidence drugs
   - Criteria: Tested in >100 cell lines, low missing data
   - Focus on drugs mentioned in reference papers (cytarabine, fludarabine, etc.)

Modeling:
1. Regression task: Predict continuous IC50 values
   - Models: Ridge, Lasso, Elastic Net, Random Forest, XGBoost
   - Evaluation: R², RMSE, Spearman correlation

2. Classification task: Sensitive vs. Resistant
   - Define threshold: Median IC50 or published cutoff
   - Models: Logistic Regression, Random Forest, SVM
   - Evaluation: ROC-AUC, Precision-Recall curves

Validation:
- 10-fold cross-validation
- External validation on GSE270494 (if drug response data available)

Deliverable:
- Notebook: "05_Drug_Response_Prediction.ipynb"
- Results table: Drug | Model | CV Performance | Top 10 CpG biomarkers
```

**Goal 2: Pathway Enrichment Analysis**
```
Tasks:
□ Annotate all significant CpGs from differential methylation analysis
□ Run GSEA using gseapy or R's clusterProfiler
□ Create publication-quality enrichment plots (bar plots, dot plots, enrichment maps)
□ Cross-reference with reference paper findings

Deliverable:
- CSV: "pathway_enrichment_results.csv"
- Visualizations: Enrichment plots for top 10 pathways
- Interpretation document: Biological meaning of findings
```

**Goal 3: Initial Deep Learning Experiment**
```
Objective: Test if deep learning improves over traditional ML

Approach:
1. Simple feedforward neural network
   - Input: Top 5K variable CpGs
   - Architecture: Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.3) → Output
   - Task: Disease classification (GSE270494) or primary site (GSE68379)

2. Compare to Random Forest baseline
   - Same train/test split
   - Same evaluation metrics

3. Analyze results:
   - Does DL outperform RF?
   - If yes, by how much? Is it worth the added complexity?
   - If no, why not? (likely: insufficient sample size)

Deliverable:
- Notebook: "06_Deep_Learning_Baseline.ipynb"
- Performance comparison table
- Decision: Proceed with DL or focus on traditional ML?
```

---

### Medium-Term Goals (Weeks 7-12)

**Goal 1: Transfer Learning for Rare Diseases** (Section 2.4)
- Implement pre-training on GSE68379
- Fine-tune on rare disease types from GSE270494
- Publish methodology paper

**Goal 2: Multi-Omics Integration** (Section 2.5)
- Download CCLE expression data
- Download COSMIC mutation data
- Build integrated predictive models
- Compare single-omics vs. multi-omics performance

**Goal 3: Interactive Dashboard Development** (Section 2.6)
- Build MVP with Streamlit
- Deploy on Heroku or AWS
- Gather feedback from collaborators
- Iterate and improve

---

### Long-Term Vision (Months 3-6)

**Research Direction 1: Novel Biomarker Panel Development**
- Goal: Identify minimal CpG panel (e.g., 10-50 sites) for clinical assay
- Approach: Recursive feature elimination + clinical validation criteria
- Target: Assay that can be run on smaller, cheaper platforms (e.g., pyrosequencing)
- Impact: Transition from research to clinical diagnostics

**Research Direction 2: Mechanistic Understanding**
- Goal: Move beyond correlation to causation
- Approach: Integrate with CRISPR screens, ChIP-seq, Hi-C data
- Question: "Do methylation changes drive drug resistance, or just correlate?"
- Validation: Propose specific experiments for wet-lab collaborators

**Research Direction 3: Expansion to Patient Samples**
- Goal: Validate cell line findings in primary tumor samples
- Data: TCGA (The Cancer Genome Atlas) methylation + clinical outcomes
- Challenge: Tumor heterogeneity, microenvironment effects
- Approach: Deconvolution methods to separate cancer cells from stroma

---

## Part 4: Recommended Technology Stack & Tools

### Core Analysis Tools
```
Data Processing:
- pandas (data manipulation)
- numpy (numerical operations)
- scipy (statistical tests)
- scikit-learn (ML algorithms, preprocessing)

Visualization:
- matplotlib (basic plots)
- seaborn (statistical visualizations)
- plotly (interactive plots)
- umap-learn (dimensionality reduction)

Bioinformatics:
- GEOparse (GEO data access) [already using]
- biopython (sequence analysis, if needed)
- gseapy (pathway enrichment)
- methylprep (450K array-specific tools, if needed)

Deep Learning:
- PyTorch or TensorFlow/Keras (neural networks)
- PyTorch Geometric (for GNN, if pursuing)
- transformers (if using attention mechanisms)

Statistics & Modeling:
- statsmodels (statistical tests, linear models)
- xgboost, lightgbm (gradient boosting)
- scikit-survival (if adding survival analysis)

Database & Deployment:
- SQLAlchemy (database ORM)
- Flask/FastAPI (web API)
- Streamlit (rapid dashboard prototyping)
- Docker (containerization)
```

### External Databases to Integrate
```
Drug Response:
- GDSC (Genomics of Drug Sensitivity in Cancer)
  URL: https://www.cancerrxgene.org
  Data: IC50 values for 453 drugs across 1000+ cell lines

- CTRP (Cancer Therapeutics Response Portal)
  URL: https://portals.broadinstitute.org/ctrp/

Gene Expression:
- CCLE (Cancer Cell Line Encyclopedia)
  URL: https://sites.broadinstitute.org/ccle/
  Data: RNA-seq for same cell lines as GSE68379

Mutations:
- COSMIC (Catalogue of Somatic Mutations in Cancer)
  URL: https://cancer.sanger.ac.uk/cosmic
  Data: Mutation calls, linked via COSMIC IDs in GSE68379

Pathway Annotations:
- MSigDB (Molecular Signatures Database)
  URL: https://www.gsea-msigdb.org/gsea/msigdb/

- KEGG, Reactome (via gseapy)

Genomic Annotations:
- Illumina 450K Manifest (CpG site annotations)
  From: Illumina website or R/Bioconductor

- ENCODE (ChIP-seq, DNase-seq, etc.)
  URL: https://www.encodeproject.org/

Patient Data (for future):
- TCGA (The Cancer Genome Atlas)
  URL: https://portal.gdc.cancer.gov/
  Data: 450K methylation + clinical outcomes for >10,000 patients
```

---

## Part 5: Key Biological Hypotheses to Test

Based on the reference papers and EDA findings, here are specific hypotheses that could be tested with AI/ML approaches:

### Hypothesis 1: Promoter Hypermethylation Predicts Nucleoside Analogue Response
**Background**: Reference paper (GSE270494) found methylation predicts response to cytarabine, fludarabine, and nelarabine (nucleoside analogues)

**Prediction**:
- CpG sites in promoters of nucleotide metabolism genes (e.g., DCK, RRM1, RRM2) will be:
  1. Differentially methylated between responders and non-responders
  2. Correlate with gene expression (if we get expression data)
  3. Generalize across GSE270494 and GSE68379 datasets

**Test**:
```python
# Pseudocode
genes_of_interest = ['DCK', 'RRM1', 'RRM2', 'CDA', 'NT5C2']
cpgs_in_promoters = filter_cpgs_by_gene_promoter(df_methylation, genes_of_interest)
drug_response = load_gdsc_data(drug='Cytarabine')
correlation = correlate(cpgs_in_promoters, drug_response)
# Expected: Negative correlation (higher methylation → lower expression → resistance)
```

**Impact**: Direct clinical relevance for treatment selection

---

### Hypothesis 2: Epigenetic Age Accelerates in Aggressive Cancers
**Background**: DNA methylation changes are used to estimate biological age (Horvath clock)

**Prediction**:
- Cell lines from aggressive cancer types (e.g., AML M7, Burkitt lymphoma) will show:
  1. Higher "epigenetic age" compared to indolent types
  2. Specific CpG sites associated with aging pathways (telomere maintenance, senescence)

**Test**:
```python
# Calculate epigenetic age using published Horvath clock CpG sites
horvath_cpgs = load_horvath_signature()  # ~350 CpG sites
epigenetic_age = estimate_age(df_methylation[horvath_cpgs])
compare_by_disease_aggressiveness(epigenetic_age, df_metadata)
```

**Impact**: Novel biomarker for disease aggressiveness

---

### Hypothesis 3: Hematological Cancers Share Methylation Signatures with Blood Cancer Cell Lines
**Background**: Both datasets contain blood cancers; GSE270494 has 180 hematological lines, GSE68379 has 177 blood-derived lines

**Prediction**:
- Overlapping cell lines will have highly correlated methylation (r > 0.9)
- Non-overlapping hematological lines from GSE270494 will cluster with blood lines from GSE68379 in PCA space
- Shared methylation signature will predict drug response across datasets

**Test**:
```python
# Find overlapping cell lines
overlap = set(gse270494_samples) & set(gse68379_blood_samples)
correlation = correlate_methylation(gse270494[overlap], gse68379[overlap])

# Test transferability of biomarkers
biomarkers_from_270494 = find_differential_cpgs(gse270494, disease='AML')
validate_in_gse68379 = test_biomarkers(gse68379_blood, biomarkers_from_270494)
```

**Impact**: Cross-study validation of biomarkers

---

### Hypothesis 4: Multi-Omics Improves Drug Response Prediction
**Background**: CCLE has expression, mutation, and methylation for same cell lines

**Prediction**:
- Methylation alone: AUC ~0.70-0.75 for drug response
- Methylation + Expression: AUC ~0.75-0.80
- Methylation + Expression + Mutations: AUC ~0.80-0.85

**Test**:
```python
# Train models with different feature combinations
model_meth = train_model(X=methylation, y=drug_response)
model_expr = train_model(X=expression, y=drug_response)
model_both = train_model(X=concat(methylation, expression), y=drug_response)

compare_auc([model_meth, model_expr, model_both])
# Hypothesis: model_both > model_meth and model_both > model_expr
```

**Impact**: Justifies multi-omics integration for precision medicine

---

## Part 6: Publication Strategy

Based on the proposed work, here are potential publication targets:

### High-Impact Computational Biology Journals

**Tier 1 (IF > 10)**
- **Nature Communications** (IF ~17)
  - Target: Deep learning architecture for methylation patterns (Section 2.2)
  - Angle: "Graph neural networks reveal 3D chromatin structure from DNA methylation"

- **Genome Biology** (IF ~12)
  - Target: Transfer learning for rare cancers (Section 2.4)
  - Angle: "Transfer learning enables drug response prediction in rare hematological malignancies"

**Tier 2 (IF 5-10)**
- **Bioinformatics** (IF ~6)
  - Target: Interactive dashboard tool (Section 2.6)
  - Angle: "PharmacoMethyl: An interactive web server for pharmacoepigenetic biomarker discovery"

- **Clinical Epigenetics** (IF ~6)
  - Target: Cross-dataset validation study (Section 2.1)
  - Angle: "Robust methylation biomarkers for nucleoside analogue response in hematological malignancies"

**Tier 3 (Domain-Specific)**
- **Nucleic Acids Research** - Web Server Issue
  - Target: Web tool with novel features
  - Angle: Tool paper for methylation explorer

- **Briefings in Bioinformatics** (IF ~9)
  - Target: Review/methods paper on pharmacoepigenomics + AI
  - Angle: "Machine learning approaches for pharmacoepigenetic biomarker discovery: A systematic review and best practices"

### Conference Presentations
- **RECOMB** (Research in Computational Molecular Biology)
- **ISMB/ECCB** (Intelligent Systems for Molecular Biology)
- **ASHG** (American Society of Human Genetics) - if clinical angle
- **NeurIPS** or **ICML** - if strong ML novelty (e.g., new GNN architecture)

---

## Part 7: Risk Assessment & Mitigation

### Technical Risks

**Risk 1: Sample Size Too Small for Deep Learning**
- **Issue**: GSE270494 has only 180 samples (71 annotated)
- **Impact**: Deep learning may overfit, perform worse than traditional ML
- **Mitigation**:
  - Use transfer learning from GSE68379 (n=1,028)
  - Data augmentation (synthetic samples via SMOTE or VAE)
  - Focus on traditional ML for GSE270494

**Risk 2: Batch Effects Between Datasets**
- **Issue**: GSE270494 and GSE68379 processed in different labs, different years
- **Impact**: Biomarkers may not transfer across datasets
- **Mitigation**:
  - Perform batch correction (ComBat, Harmony)
  - Use only common CpG sites (~485K overlap)
  - Validate in independent dataset (e.g., TCGA)

**Risk 3: Drug Response Data Missing or Low Quality**
- **Issue**: GDSC may not have data for all cell lines or drugs
- **Impact**: Limited samples for supervised learning
- **Mitigation**:
  - Focus on high-confidence drugs (tested in >100 lines)
  - Use imputation methods for missing values
  - Alternative: Use drug class (e.g., alkylating agents) instead of individual drugs

### Biological Risks

**Risk 4: Methylation May Not Predict Drug Response**
- **Issue**: Drug response driven by mutations, not methylation
- **Impact**: Null results, no significant biomarkers
- **Mitigation**:
  - Check literature: Reference papers already show methylation associations
  - Multi-omics approach: Even if methylation alone weak, combination may work
  - Reframe: "When does methylation matter?" (specific drug classes or pathways)

**Risk 5: Cell Lines Don't Represent Patient Tumors**
- **Issue**: Cell lines are immortalized, lack microenvironment
- **Impact**: Biomarkers won't translate to clinic
- **Mitigation**:
  - Validate in TCGA patient samples (future work)
  - Acknowledge limitation in paper
  - Position as "hypothesis generation" for experimental validation

### Computational Risks

**Risk 6: Insufficient Computational Resources for Deep Learning**
- **Issue**: Local machine may not handle large models or datasets
- **Impact**: Delays in experiments, cannot pursue some directions
- **Mitigation**:
  - Start with smaller models (1D-CNN on subsets)
  - Use free cloud resources (Google Colab with GPU, Kaggle kernels)
  - Apply for academic cloud credits (AWS, Google Cloud, Azure)

---

## Part 8: Success Metrics

### Quantitative Metrics

**Data Analysis:**
- [ ] Identified >100 differentially methylated CpG sites per disease type (FDR < 0.05)
- [ ] Found >50 CpG sites associated with drug response (p < 0.001)
- [ ] Cross-dataset correlation for overlapping samples: r > 0.85

**Predictive Modeling:**
- [ ] Disease classification: Accuracy > 0.80, AUC > 0.85 (10-fold CV)
- [ ] Drug response prediction: Spearman r > 0.40 (literature benchmark: r ~0.3-0.5)
- [ ] Transfer learning: Performance gain of >10% vs. training from scratch

**Biological Validation:**
- [ ] >10 enriched pathways (FDR < 0.05) that match literature
- [ ] >50% of top CpG biomarkers in known cancer genes or regulatory regions
- [ ] At least 3 testable hypotheses for experimental validation

### Qualitative Metrics

**Scientific Impact:**
- [ ] At least 1 manuscript submitted to peer-reviewed journal
- [ ] Novel finding not reported in reference papers
- [ ] Collaboration initiated with experimental lab for validation

**Tool Development:**
- [ ] Functional web dashboard deployed and publicly accessible
- [ ] Code repository on GitHub with >10 stars
- [ ] Tutorial/documentation enabling others to reproduce analysis

**Learning & Growth:**
- [ ] Mastery of pharmacoepigenomics domain
- [ ] Implementation of at least 1 new ML technique (e.g., GNN, transfer learning)
- [ ] Established workflow for multi-omics integration

---

## Part 9: Recommended Resources for Deep Dive

### Key Papers to Read

**Pharmacoepigenomics Foundations:**
1. Original papers in references/ folder (GSE270494 and GSE68379 publications)
2. "Pharmacoepigenomics" - review in Nature Reviews Drug Discovery
3. "DNA methylation and drug resistance" - Cancer Research reviews

**Machine Learning for Genomics:**
1. "Deep learning for genomics: A concise overview" - Nature Reviews Genetics (2019)
2. "Graph neural networks for molecular and genomic data" - arXiv
3. "Transfer learning in biomedical natural language processing" - Bioinformatics

**Methylation Analysis Methods:**
1. "Tutorial: Guidelines for the computational analysis of DNA methylation" - Nature Protocols
2. "ChAMP: 450K Chip Analysis Methylation Pipeline" - Bioinformatics
3. "Minfi: A flexible and comprehensive Bioconductor package for the analysis of Infinium DNA methylation microarrays" - Bioinformatics

### Online Courses & Tutorials
- **Coursera**: "Bioinformatics Specialization" (UC San Diego)
- **edX**: "Data Science for Genomics"
- **YouTube**: StatQuest (Josh Starmer) - ML and statistics explanations
- **Kaggle**: "Learn" section for pandas, scikit-learn tutorials

### Communities to Join
- **Bioinformatics Stack Exchange** - Q&A for computational biology
- **r/bioinformatics** (Reddit) - community discussions
- **ResearchGate** - connect with paper authors
- **Twitter/X** - follow #bioinformatics, #CompBio, #MachineLearning

---

## Conclusion & Next Session Planning

### Summary of Opportunities

We are at an exciting juncture with two high-quality DNA methylation datasets that offer multiple pathways to impactful research:

1. **Immediate Impact**: Cross-dataset validation of pharmacoepigenetic biomarkers (3-6 weeks)
2. **Methodological Innovation**: Transfer learning for rare cancers (6-12 weeks)
3. **Tool Development**: Interactive methylation explorer (8-12 weeks)
4. **High-Risk/High-Reward**: Deep learning architectures for spatial methylation patterns (3-6 months)

The data quality is excellent, the biological questions are well-motivated, and existing literature provides strong validation benchmarks. The key decision is **which direction to pursue first** based on:
- Your background and interests (pure ML vs. biological discovery vs. tool building)
- Available computational resources
- Timeline for first publication

### Recommended Starting Point

**My recommendation: Start with Section 2.1 (Cross-Dataset Biomarker Discovery)**

**Rationale:**
- ✅ Builds directly on completed EDA
- ✅ Uses established ML methods (lower risk)
- ✅ Produces interpretable results (CpG biomarkers + biological validation)
- ✅ Publishable in 3-6 months (realistic timeline)
- ✅ Establishes foundation for more complex work (deep learning, multi-omics)

**Concrete Next Steps for Tomorrow:**
```
1. Run differential methylation analysis for top 3 disease types in GSE270494
   (AML, DLBCL, B-ALL) - start with AML vs. all others

2. Download GDSC drug response data and match to GSE68379 samples

3. Create baseline predictive model (Random Forest) for:
   - Disease classification (GSE270494)
   - Drug response for 1 drug (e.g., cytarabine)

4. If results promising → proceed to full cross-dataset validation study
   If results weak → pivot to multi-omics integration (Section 2.5)
```

### Questions for Team Discussion

Before proceeding, consider discussing:
1. **Primary goal**: Publication? Tool development? Learning ML techniques?
2. **Collaboration potential**: Access to wet-lab for validation experiments?
3. **Computational resources**: GPU access for deep learning?
4. **Timeline**: Aiming for publication in 6 months? 12 months?
5. **Risk tolerance**: Safe path (traditional ML) vs. high-risk/high-reward (novel architectures)?

---

**This findings document will be updated iteratively as new analyses are completed. Version 1.0 reflects state as of October 20, 2025.**

---

## Appendix: Code Snippets for Quick Start

### A1: Differential Methylation Analysis (Disease Types)

```python
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Load data
df_meth = pd.read_csv('../data/processed/GSE270494_methylation_top10k_variable.csv.gz',
                       index_col=0, compression='gzip')
df_meta = pd.read_csv('../data/processed/GSE270494_sample_metadata.csv', index_col=0)

# Compare AML vs. non-AML
aml_samples = df_meta[df_meta['disease_type'] == 'AML'].index
non_aml_samples = df_meta[(df_meta['disease_type'] != 'AML') &
                           (df_meta['disease_type'] != 'Unknown')].index

# T-test for each CpG site
pvalues = []
foldchanges = []

for cpg in df_meth.index:
    aml_vals = df_meth.loc[cpg, aml_samples]
    non_aml_vals = df_meth.loc[cpg, non_aml_samples]

    t_stat, p_val = stats.ttest_ind(aml_vals, non_aml_vals, nan_policy='omit')
    pvalues.append(p_val)

    fc = aml_vals.mean() - non_aml_vals.mean()
    foldchanges.append(fc)

# Multiple testing correction
reject, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')

# Create results dataframe
df_results = pd.DataFrame({
    'CpG': df_meth.index,
    'pvalue': pvalues,
    'FDR': pvals_corrected,
    'MeanDiff': foldchanges,
    'Significant': reject
})

df_results = df_results.sort_values('FDR')
print(f"Significant CpG sites (FDR < 0.05): {reject.sum()}")
df_results.head(20)
```

### A2: Download GDSC Drug Response Data

```python
# GDSC provides drug response via web API or bulk download
# Option 1: Use their API (example for one drug)

import requests
import pandas as pd

# Get cell line annotations
url_cell_lines = 'https://www.cancerrxgene.org/api/celllines'
response = requests.get(url_cell_lines)
cell_lines = pd.DataFrame(response.json())

# Get drug response (example: Cytarabine, Drug ID: 1003)
drug_id = 1003  # Cytarabine
url_drug = f'https://www.cancerrxgene.org/api/compounds/{drug_id}/sensitivity'
response = requests.get(url_drug)
drug_response = pd.DataFrame(response.json())

# Match to your cell lines
merged = pd.merge(
    drug_response,
    cell_lines,
    left_on='cell_line_id',
    right_on='id'
)

print(f"Cell lines with {drug_name} response: {len(merged)}")
```

### A3: Baseline Classification Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare data (disease classification example)
X = df_meth.T  # Samples as rows, CpGs as columns
y = df_meta.loc[X.index, 'disease_type']

# Remove Unknown samples
mask = y != 'Unknown'
X = X[mask]
y = y[mask]

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"10-Fold CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Train final model for inspection
model.fit(X, y)

# Feature importance
feature_importance = pd.DataFrame({
    'CpG': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important CpG sites:")
print(feature_importance.head(10))
```

### A4: Pathway Enrichment Analysis

```python
import gseapy as gp

# Get genes associated with significant CpGs
# (Requires Illumina 450K annotation file - download from Illumina or Bioconductor)

# Example: Load annotation (you'll need to download this)
# anno = pd.read_csv('path/to/HumanMethylation450_15017482_v1-2.csv', skiprows=7)
# anno = anno[['IlmnID', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group']]

# Map significant CpGs to genes
# significant_cpgs = df_results[df_results['Significant']]['CpG'].tolist()
# genes = anno[anno['IlmnID'].isin(significant_cpgs)]['UCSC_RefGene_Name'].dropna()
# gene_list = set([g for genes_str in genes for g in genes_str.split(';')])

# For demonstration, let's use example gene list
gene_list = ['TP53', 'BRCA1', 'EGFR', 'KRAS']  # Replace with your genes

# Run GSEA
enr = gp.enrichr(
    gene_list=gene_list,
    gene_sets='KEGG_2021_Human',  # or 'GO_Biological_Process_2021'
    organism='human',
    outdir=None,
    cutoff=0.05
)

# View results
print(enr.results.head(10))

# Plot
gp.barplot(enr.res2d, title='KEGG Pathway Enrichment')
```

---

**Document Status**: Version 1.0 - Initial Findings Report
**Last Updated**: October 20, 2025
**Next Review**: After completing Priority 1-3 immediate next steps
