# Decision Log: PRISM Pharmacoepigenomics Project

**Purpose**: Chronological record of every analytical, architectural, and methodological decision made during this project, for manuscript Methods/Supplementary drafting.

**Started**: October 20, 2025
**Last Updated**: February 26, 2026 (Session 8, continued)

---

## How to Use This Document

Each entry records: **what** was decided, **why** (rationale), **what alternatives** were considered, and **where** in the codebase/docs the decision is implemented. Entries are grouped by project phase and ordered chronologically within each phase.

---

## Phase 1: Exploratory Data Analysis (Oct 20-22, 2025)

### D-001: Primary Dataset Selection -- GSE68379
**Date**: Oct 20, 2025 (Session 1/4)
**Decision**: Use GSE68379 (1,028 pan-cancer cell lines, Illumina 450K) as the primary training dataset. Use GSE270494 (180 hematological malignancy cell lines) as a secondary/validation dataset.
**Rationale**: GSE68379 covers 22 cancer types with 485K CpG sites and links directly to GDSC drug response and gene expression data. GSE270494 is narrower (hematological only) but useful for cross-dataset validation.
**Alternatives considered**: Using GSE270494 alone (too narrow), DepMap/CCLE methylation (download unsuccessful), TCGA (no IC50 labels).
**Citations**: Iorio et al. (2016) Cell 166(3):740-754 (GSE68379); Noguera-Castells et al. (2025) Leukemia (GSE270494).
**Implemented in**: `data/raw/GSE68379/`, `data/raw/GSE270494/`

### D-002: Methylation Platform -- Illumina 450K
**Date**: Oct 20, 2025 (Session 1)
**Decision**: Use Illumina HumanMethylation450K array data (~485K CpG sites) as the methylation measurement platform.
**Rationale**: Both primary datasets (GSE68379, GSE270494) use this platform. It is the most widely profiled platform in GDSC cell lines. CpGPT foundation model (selected later) was trained on 450K arrays natively.
**Alternatives considered**: EPIC 850K array (fewer samples available), WGBS (not available for these cell lines).

### D-003: Beta-Value Filtering -- Remove Detection P-values
**Date**: Oct 20, 2025 (Session 2)
**Decision**: Automatically filter out detection p-value columns from GSE270494 data, keeping only beta-value columns. Implemented pattern-matching on column names containing "Detection Pval".
**Rationale**: Raw data interleaves beta-values and detection p-values (360 columns for 180 samples). Initial statistics were incorrect because p-value columns were included in calculations (mean=0.30 vs correct mean=0.59).
**Implemented in**: `src/data/loading.py:219-227`, `extract_methylation_matrix()` function

### D-004: Feature Selection -- Top 10K Most Variable CpG Sites
**Date**: Oct 20, 2025 (Session 2)
**Decision**: Select the 10,000 CpG sites with highest variance across cell lines for dimensionality reduction and downstream ML.
**Rationale**: 485K CpG sites is too many features for 987 samples (p >> n). High-variance sites are most likely to contain biologically informative signal (tissue-of-origin, disease state). Variance range of selected sites: 0.13-0.22. This 50:1 reduction preserves discriminative power while making ML tractable.
**Alternatives considered**: All 485K sites (memory-prohibitive, p >> n), PCA directly on full matrix (3.7 GB RAM insufficient), fewer features (5K too aggressive), more features (50K still p >> n).
**Implemented in**: `notebooks/reference/03_Prepare_ML_Ready_Dataset.ipynb`, `data/processed/ML_dataset_methylation_features.csv.gz`

### D-005: Standardization Before PCA -- StandardScaler
**Date**: Oct 20, 2025 (Session 2)
**Decision**: Apply StandardScaler (zero mean, unit variance) to CpG beta-values before PCA.
**Rationale**: PCA is sensitive to feature scale. Although beta-values are bounded [0,1], their variances differ by orders of magnitude. Standardizing ensures high-variance sites don't dominate the first principal components purely due to scale.
**Implemented in**: Notebooks, PCA analysis sections

### D-006: Reproducibility Seed -- 42
**Date**: Oct 20, 2025 (Session 1)
**Decision**: Use random seed = 42 for all random operations (numpy, sklearn, torch, python random).
**Rationale**: Standard convention. Ensures reproducibility across runs and machines.
**Implemented in**: All scripts and notebooks, `.claude/context/coding_standards.md`

### D-007: Clustering Parameters
**Date**: Oct 20, 2025 (Session 3)
**Decision**: Hierarchical clustering with Ward linkage, Euclidean distance, Z-score normalization, using top 1,000 most variable CpG sites for heatmap visualization.
**Rationale**: Ward linkage minimizes within-cluster variance (standard for gene expression/methylation). Top 1K sites balance signal density with visual readability in heatmaps. Z-score normalization makes cross-CpG comparisons fair.
**Implemented in**: `src/visualization/plot_gse270494.py`, EDA notebooks

### D-008: Data Download Strategy -- Processed Matrix Only
**Date**: Oct 20, 2025 (Session 4)
**Decision**: Download only the processed beta-value matrix from GEO, not individual sample IDATs or raw files.
**Rationale**: GEOparse attempted to download 1,000+ individual sample files (~4 GB unnecessary). The processed matrix is pre-normalized and sufficient for our analysis. Saved ~2-3 hours of download time.
**Alternatives considered**: Download all raw IDATs and reprocess (unnecessary for our purposes), use GEOparse automated download (too slow, excessive data).

### D-009: Sample Metadata Annotation -- Literature-Based
**Date**: Oct 20, 2025 (Session 2)
**Decision**: For GSE270494, annotate cell line disease types by matching cell line names against literature-curated disease mappings (71/180 annotated = 39%). For GSE68379, use GEO metadata annotations directly (1,028/1,028 = 100%).
**Rationale**: GSE270494 GEO SOFT files only contain generic "Hematologic cell line" annotation. Manual curation against published cell line databases (DSMZ, ATCC) required.

---

## Phase 2: ML Dataset Preparation (Oct 24, 2025)

### D-010: Research Direction -- Baseline ML + Biomarker Discovery
**Date**: Oct 24, 2025 (Session 6)
**Decision**: Start with cross-dataset biomarker discovery: train baseline ML models (RF, XGBoost) for tissue/drug response classification, then identify predictive CpG biomarkers.
**Rationale**: Low risk, publishable in 3-6 months, builds foundation for more ambitious GNN work. Group consensus.
**Alternatives considered**: 7 directions identified in findings.md -- deep learning (higher risk), multi-omics integration (requires more data), web dashboard (community tool), transfer learning, pathway enrichment, multi-drug analysis.
**Documented in**: `notebooks/reference/findings.md` Section 2.1-2.7

### D-011: ML Dataset Format -- CSV.gz
**Date**: Oct 24, 2025 (Session 6)
**Decision**: Store ML-ready datasets as gzip-compressed CSV files (`.csv.gz`).
**Rationale**: Universally compatible with pandas, R, Julia. No special libraries needed to read. Compression reduces 77.7 MB dataset to manageable size. Groupmate can load immediately.
**Alternatives considered**: Parquet (faster loading, smaller, but less familiar to groupmate), HDF5 (overkill for tabular data), uncompressed CSV (too large for git-adjacent workflows).

### D-012: Drug Response Source -- GDSC1
**Date**: Oct 24, 2025 (Session 6)
**Decision**: Use GDSC1 fitted dose-response IC50 values (release 8.5, Oct 2023) as drug response labels.
**Rationale**: GDSC1 is the most widely benchmarked drug response dataset in the field. IC50 (half-maximal inhibitory concentration) is the standard metric. Release 8.5 is the latest available.
**Alternatives considered**: GDSC2 (different drug panel, less benchmarked), CTRPv2 (recommended by IMPROVE framework but different cell line overlap), AUC metric (less interpretable than IC50).
**Data file**: `data/raw/GDSC/GDSC1_fitted_dose_response_27Oct23.xlsx`

### D-013: Drug Filtering -- Minimum 100 Cell Lines Tested
**Date**: Oct 24, 2025 (Session 6)
**Decision**: Include only drugs tested in >= 100 cell lines in GDSC1, yielding 375 drugs.
**Rationale**: Drugs tested in fewer cell lines provide insufficient training signal. The 100-cell-line threshold balances drug count (375 is substantial) with per-drug sample size. This matches standard practice in the GDSC benchmarking literature.
**Implemented in**: `integrate_gdsc_data.py`, `data/processed/available_drugs.csv`

### D-014: Gene Expression Integration
**Date**: Oct 2025 (Session 6-7)
**Decision**: Integrate GDSC RMA-normalized gene expression (~19K genes) with methylation features. Final dataset: 987 cell lines with both methylation (10K CpGs) and expression (19K genes) plus 375 drug IC50 values.
**Rationale**: Multi-omics (methylation + expression) provides complementary information. Methylation captures gene silencing; expression captures active transcription. Both are needed for the CpG-Gene graph architecture.
**Data file**: `data/processed/ml_with_gene_expr.csv.gz` (987 x 29,265)

---

## Phase 3: Flat-Vector Baselines (Oct-Nov 2025)

### D-015: Baseline Models -- RF, XGBoost, Lasso, GAM
**Date**: Oct-Nov 2025 (Sessions 6-7)
**Decision**: Train Random Forest, XGBoost, Lasso regression, and Generalized Additive Models as flat-vector baselines for drug response prediction.
**Rationale**: Establish performance floor before investing in complex architectures. These are standard, well-understood models that require no GPU.
**Key results**:
- Random split: R^2 ~0.05-0.10 (single drug, Avagacestat)
- Histology-blind split: R^2 = -0.06 (worse than mean prediction)
- Site-blind split: R^2 = -0.02 (worse than mean prediction)
**Conclusion**: Flat-vector methods fail on generalization splits. This motivates graph-based approaches that can capture biological structure.
**Implemented in**: `model_selections.ipynb`, `harder_ds.ipynb`, `multitask_drug_response.ipynb`

### D-016: Evaluation Splits -- Random + Histology-Blind + Site-Blind
**Date**: Oct-Nov 2025 (Sessions 6-7)
**Decision**: Evaluate all models on three split strategies: (1) random 70/15/15 train/val/test, (2) histology-blind (leave-one-histology-out), (3) site-blind (leave-one-primary-site-out).
**Rationale**: Random splits overestimate generalization because similar cell lines appear in both train and test. Histology-blind and site-blind splits test whether the model generalizes to truly unseen cancer types -- the clinically relevant scenario. This matches prior work (TransCDR cold-cell evaluation).
**Alternatives considered**: Leave-one-cell-line-out (too many folds), k-fold CV only (doesn't test hard generalization), tissue-specific models (too few samples per tissue).

### D-017: Train/Val/Test Ratio -- 70/15/15
**Date**: Oct 2025
**Decision**: Use 70% training, 15% validation, 15% test split ratio.
**Rationale**: With 987 samples, this gives ~690 train, ~148 val, ~149 test. Sufficient for training while retaining meaningful test sets. Stratified by primary site when possible. Standard in the field.
**Implemented in**: Coding standards (`.claude/context/coding_standards.md`)

---

## Phase 4: Architecture & Strategy (Feb 7-26, 2026)

### D-018: Literature Review -- Novelty Gap Confirmed
**Date**: Feb 7, 2026 (Session 7)
**Decision**: Confirmed that no published method models CpG-gene regulatory relationships as graph structure for drug response prediction, as of Feb 2026.
**Evidence**:
- GraphAge (PNAS Nexus 2025): CpG graphs for age prediction, not drug response
- GraphMeXplain (bioRxiv Jan 2026): CpG graphs for methylation analysis, not drug response
- All drug response GNNs (GraphTCDR, HGACL-DRP, DRPreter, DeepCDR, TransCDR): use gene-only or drug-only graphs, or methylation as flat vectors
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 1.1

### D-019: Architecture Selection -- CpG-Gene Heterogeneous Graph (Architecture B)
**Date**: Feb 7, 2026 (Session 7)
**Decision**: Pursue Architecture B (CpG-Gene heterogeneous graph with HGT encoder) as the primary contribution, with Architecture A (GAT on STRING PPI) as baseline.
**Rationale**: Architecture B is the novelty claim -- first heterogeneous graph connecting CpG methylation nodes to gene expression nodes via regulatory edges for drug response prediction. Architecture A validates infrastructure.
**Alternatives considered**:
- Architecture A only (GAT on STRING PPI): solid but not novel enough
- Architecture C (pathway-hierarchical GPS + contrastive): too complex for initial paper, elements can be added incrementally
**Progression**: A (weeks 3-4) -> B (weeks 5-7) -> add C elements based on ablation results
**Documented in**: `docs/GNN_Architecture_Proposals.md` (2,147 lines, full code)

### D-020: Graph Structure -- 3 Edge Types
**Date**: Feb 7, 2026 (Session 7)
**Decision**: Build heterogeneous graph with 3 edge types:
1. **CpG -> Gene**: From Illumina 450K manifest (TSS proximity, <1500bp)
2. **Gene <-> Gene**: From STRING v12 PPI (human, confidence > 700)
3. **CpG <-> CpG**: Co-methylation or shared-gene annotation (optional, added in ablation)
**Rationale**: Each edge type encodes real regulatory biology. CpG->Gene edges represent epigenetic regulation (methylation silences nearby genes). Gene<->Gene edges represent protein-protein interactions and pathway membership. The graph is static (same structure for all cell lines); what changes per cell line is the node feature values.
**Expected sizes**: ~10K CpG nodes, ~5K gene nodes, ~15-30K CpG->Gene edges, ~50-200K Gene<->Gene edges.
**Fallback**: If >30% of CpG nodes have zero gene edges, expand TSS distance threshold from 1500bp to 5000bp.

### D-021: GNN Encoder -- HGT (Heterogeneous Graph Transformer)
**Date**: Feb 7, 2026 (Session 7)
**Decision**: Use HGT (Heterogeneous Graph Transformer) from PyG as the graph encoder. 2-3 layers, 128 hidden dim, 4-8 attention heads.
**Rationale**: HGT is the standard encoder for heterogeneous graphs in PyG. Per-node-type linear projections naturally handle CpG vs gene nodes having different feature dimensions. Multi-head attention across edge types allows the model to learn which regulatory relationships matter most.
**Alternatives considered**: RGCN (simpler but no attention), HAN (hierarchical attention), custom GAT (more engineering effort for same result).

### D-022: Foundation Model -- scGPT for Gene Expression
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use scGPT (pan-cancer checkpoint, 51M params, 512-dim cell embedding) as the gene expression encoder. Frozen, not fine-tuned.
**Rationale**: scGPT won zero-shot evaluation in scDrugMap benchmark (Nature Communications 2025). Has direct drug response validation via DeepCDR integration (arXiv Apr 2025). Simpler input format than Geneformer (no rank-value encoding). Smaller model (51M vs 316M for Geneformer V2).
**Alternatives considered**:
- Geneformer V2 with CLcancer checkpoint (larger, continual-learned on 14M cancer transcriptomes, Apache 2.0) -- kept as fallback
- CancerFoundation (smaller, cancer-specific) -- via Helical API for quick benchmarking
- UCE (won after fine-tuning in scDrugMap) -- requires fine-tuning
- PCA on raw expression (strong baseline, always included for comparison)
**Risk**: scGPT is a single-cell model applied to bulk cell line data. Mitigate by always comparing against PCA baseline.
**Citation**: Cui et al., Nature Methods 2024 (scGPT); scDrugMap: Nature Communications 2025
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 2.1

### D-023: Foundation Model -- CpGPT for DNA Methylation
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use CpGPT-100M as the DNA methylation encoder. Frozen, not fine-tuned.
**Rationale**: First true methylation foundation model. 100M params, trained on 150K+ samples from 2000+ datasets. Supports 450K arrays natively (exact domain match for our data). Trained on bulk data (not single-cell), so no domain gap. MIT license.
**Alternatives considered**:
- MethylGPT (15M params, 256-dim, Apache 2.0) -- kept as fallback, faster but smaller
- PCA on top-N variable CpGs (strong baseline per Genome Biology 2025) -- always included for comparison
**Fallback**: If CpGPT installation fails, use MethylGPT or PCA.
**Citation**: Camillo et al., bioRxiv Oct 2024 (github.com/lcamillo/CpGPT)
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 2.2

### D-024: Drug Encoding -- ECFP-4 Fingerprints (2048-bit)
**Date**: Feb 26, 2026 (Session 8)
**Decision**: Use ECFP-4 fingerprints (Morgan fingerprints, radius=2, 2048 bits) as the primary drug encoding. One-hot (375-dim identity) as ablation baseline. No neural drug encoders.
**Rationale**: A rigorous benchmark of 25 pretrained molecular models (arXiv:2508.06199, Aug 2025) found that nearly all neural molecular encoders show negligible improvement over ECFP fingerprints. Top performers (CLAMP, R-MAT, MolBERT) offered marginal gains. GNN-based drug encoders ranked worst. This means investing in Uni-Mol2, ChemBERTa, or heavy neural drug encoders is not worth the complexity.
**Implementation**: RDKit `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)` for each drug with valid SMILES.
**Alternatives considered**: Uni-Mol2 (1.1B params, overkill), ChemBERTa (marginal gains), pre-trained GIN from DeepCDR/TransCDR (optional addition later), learned embeddings (risk of overfitting with 375 drugs).
**Citation**: arXiv:2508.06199 (Aug 2025) -- molecular embedding benchmark
**Implemented in**: `encode_drugs.py`

### D-025: Foundation Model Freezing -- No Fine-Tuning
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Keep all foundation models (scGPT, CpGPT) frozen during training. Extract embeddings once, save to disk, train only the GNN + fusion head.
**Rationale**: 987 samples is too few to fine-tune 51M+ parameter models without severe overfitting. Freezing saves ~90% of compute budget. If frozen embeddings clearly fail (ablation #2: raw features vs FM), reconsider with LoRA fine-tuning as a stretch goal.
**Documented in**: `docs/Development_Plan.md` Section 7, `docs/Refined_Strategy_Feb2026.md` Section 4.2

### D-026: Training Strategy -- Multi-Task (All 375 Drugs)
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Train a single model to predict IC50 for all 375 drugs simultaneously (multi-task), rather than 375 separate per-drug models.
**Rationale**: With only 987 cell lines, per-drug models have extremely limited training data (as few as 100 samples per drug after splitting). Multi-task learning provides implicit regularization through the shared encoder. The drug encoding vector (ECFP or one-hot) conditions the output for each drug.
**Alternatives considered**: Per-drug models (insufficient data per drug), drug-cluster models (arbitrary grouping), sequential multi-task with curriculum learning (unnecessary complexity at this stage).

### D-027: Cell-Drug Fusion -- Bilinear Attention
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use bilinear attention as the primary fusion mechanism for combining cell embeddings and drug embeddings to predict IC50.
**Rationale**: Bilinear attention captures multiplicative interactions between cell and drug representations (e.g., "this pathway is active AND this drug targets that pathway"). More expressive than simple concatenation.
**Alternatives considered**: Tensor product (tested in ablation), simple concatenation + MLP (ablation baseline), cross-attention (too complex for initial implementation).
**Documented in**: `docs/Development_Plan.md` Section 3.1

### D-028: Compute Budget -- 1000 A100 Hours on Delta
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Allocate ~1000 A100 GPU hours on NCSA Delta (NSF ACCESS) across 6 phases:
- Phase 0 (embedding extraction): 5-15 hrs
- Phase 1 (tabular baselines): 0 hrs (CPU)
- Phase 2 (Architecture A): 50-100 hrs
- Phase 3 (Architecture B): 200-400 hrs
- Phase 4 (cross-validation): 100-200 hrs
- Phase 5 (ablations): 100-200 hrs
- Buffer: 85-545 hrs
**Rationale**: 987 samples is small enough that single-GPU training is sufficient. BF16 mixed precision on A100 gives 2x throughput. Tabular baselines require zero GPU (CPU-only). Embedding extraction is one-time cost.
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 4

### D-029: Ablation Study Design -- 8-10 Experiments
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Run 8-10 ablation experiments to prove each component contributes:
1. Full model (reference)
2. Raw features instead of FM embeddings (value of pre-training)
3. Gene-only graph, no CpG nodes (value of methylation graph)
4. CpG-only graph, no gene nodes (value of expression graph)
5. No graph, flat concat of embeddings -> MLP (value of graph structure)
6. Random edges, same density (biology vs regularization)
7. Remove CpG->Gene edges only (value of regulatory connections)
8. Remove Gene<->Gene edges only (value of PPI network)
9. One-hot drug encoding instead of ECFP (value of molecular structure)
10. Single-task per-drug instead of multi-task (value of multi-task learning)
Each ablation with 3-5 random seeds.
**Rationale**: Ablations are required for credible publication. Each isolates one component's contribution. Random-edges ablation (#6) is critical -- if random edges match real edges, the graph provides only regularization, not biological signal.
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 3.3, `docs/Development_Plan.md` Phase 6

### D-030: Benchmark Targets
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Primary benchmark target is TransCDR's cold-cell PCC of 0.864. Secondary benchmarks: DeepCCDS (PCC 0.93 random), GraphTCDR (SOTA on PRISM), HGACL-DRP (AUC 0.99).
**Rationale**: TransCDR is the strongest published generalization result from a method that also uses methylation. Beating it on cold-cell (unseen cell lines) would be a strong result. Random-split performance is secondary because it doesn't test generalization.
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 1.2

### D-031: Novelty Claims -- Three Defined
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Paper will make three novelty claims:
1. **Primary**: First heterogeneous GNN encoding CpG-gene regulatory relationships for cancer drug response prediction.
2. **Secondary**: First application of DNA methylation foundation model (CpGPT) embeddings as node features in a drug response GNN.
3. **Tertiary**: Demonstration that graph-structured epigenetic-transcriptomic relationships improve generalization to unseen cancer types where flat-vector methods fail.
**Rationale**: Claim 1 is the core structural novelty (confirmed gap in literature). Claim 2 adds methodological novelty (CpGPT is new, never applied to DRP). Claim 3 is the practical contribution (clinically relevant generalization).
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 6

### D-032: Fallback Strategy -- If Graph Doesn't Help
**Date**: Feb 17, 2026 (Session 8)
**Decision**: If ablations show flat concatenation matches the GNN, pivot paper framing to "foundation model embeddings (CpGPT + scGPT) for drug response" as the contribution, with attention-based fusion instead of graph message passing.
**Rationale**: The embeddings themselves are novel for this application. A negative result (graph doesn't help) is still publishable if the foundation model embeddings beat prior work on hard splits.
**Documented in**: `docs/Development_Plan.md` Section 3.3

---

## Phase 5: GNN Implementation (Feb 26, 2026 - ongoing)

### D-033: Data Pipeline Script -- download_all_data.py
**Date**: Feb 26, 2026 (Session 8)
**Decision**: Create a single comprehensive download script (`download_all_data.py`) that downloads all 9 raw data sources needed for the GNN pipeline, with resume support, validation, and rate-limited PubChem API calls.
**Rationale**: Raw data was only on local machine, not on Delta HPC. A single reproducible script ensures anyone can reconstruct the data directory from scratch. Resume support (via JSON cache for SMILES, partial file detection for downloads) handles network interruptions common on HPC.
**Data sources**: GSE68379 methylation (3.9 GB), GDSC expression (137 MB), GDSC1 drug response (28 MB), GDSC compounds CSV, GDSC cell line annotations, STRING v12 PPI (79 MB), STRING v12 protein info (1.9 MB), Illumina 450K manifest (192 MB), drug SMILES from PubChem API.
**Implemented in**: `download_all_data.py`

### D-034: SMILES Coverage -- 429/542 GDSC Compounds (79%)
**Date**: Feb 26, 2026 (Session 8)
**Decision**: Accept 79% SMILES coverage (429/542 GDSC compounds). The 113 missing are mostly internal screening codes unlikely to be in our 375-drug modeled set. Drugs without SMILES use one-hot encoding as fallback.
**Rationale**: PubChem lookup used multi-variant name matching (CID, exact name, desalted, no-hyphens, parenthetical aliases, GDSC synonyms). The missing 113 drugs have names like screening codes that aren't in public chemical databases. Coverage on the 375 modeled drugs is expected to be higher.
**Implemented in**: `download_all_data.py` (SMILES download), `check_smiles_coverage.py`, `check_smiles_impact.py`

### D-035: Drug Encoding Script -- encode_drugs.py
**Date**: Feb 26, 2026 (Session 8)
**Decision**: Create `encode_drugs.py` to generate ECFP-4 fingerprints, one-hot encodings, drug index mapping, and 4 publication-quality figures visualizing the drug landscape.
**Outputs**:
- `embeddings/drug_ecfp.npy` -- (N, 2048) ECFP-4 fingerprints
- `embeddings/drug_onehot.npy` -- (375, 375) identity matrix
- `embeddings/drug_index.csv` -- drug name to index mapping with metadata
- 4 figures: t-SNE chemical space, Tanimoto similarity heatmap, dataset overview dashboard, pathway coverage chart
**Rationale**: ECFP-4 fingerprints are the confirmed drug encoding for the GNN pipeline. The figures validate the drug data pipeline and provide publication-ready visualizations for manuscript Figure 1.
**Implemented in**: `encode_drugs.py`

### D-036: Tanimoto Similarity -- Vectorized Numpy
**Date**: Feb 26, 2026 (Session 8)
**Decision**: Compute pairwise Tanimoto similarity using vectorized numpy operations (dot product for intersection, addition for union) rather than RDKit's `DataStructs.TanimotoSimilarity`.
**Rationale**: For N drugs, pairwise comparison requires N^2/2 operations. The vectorized approach computes the full matrix in one step via matrix multiplication, which is orders of magnitude faster than iterating with RDKit's pairwise function.
**Implementation**: `intersection = bits @ bits.T; union = counts[:, None] + counts[None, :] - intersection; tanimoto = intersection / union`
**Implemented in**: `encode_drugs.py`, `compute_tanimoto_matrix()` function

### D-037: available_drugs.csv -- Git-Tracked Exception
**Date**: Feb 26, 2026 (Session 8)
**Decision**: Add `!data/processed/available_drugs.csv` exception to `.gitignore` so this small file is tracked in git and available on Delta after `git pull`.
**Rationale**: `encode_drugs.py` needs this file to know which 375 drugs are in the modeled set. It's small (<10 KB) and doesn't contain sensitive data. The blanket `data/**/*` and `*.csv` rules would otherwise exclude it.
**Implemented in**: `.gitignore` (line 16)

---

## External Data Sources

### D-038: STRING v12 PPI Network
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use STRING v12 human PPI network (organism 9606) with confidence threshold > 700 for Gene<->Gene edges.
**Rationale**: STRING is the most comprehensive curated PPI database. Confidence > 700 ("high confidence") filters out predicted/low-evidence interactions while retaining ~500K experimentally validated or text-mined edges. v12 is the latest release.
**Source**: stringdb-downloads.org, `9606.protein.links.v12.0.txt.gz` (~79 MB)
**Citation**: Szklarczyk et al., Nucleic Acids Research 2023

### D-039: Illumina 450K Manifest for CpG->Gene Edges
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use the Illumina HumanMethylation450K manifest to map CpG probes to nearest genes (TSS proximity, <1500bp) for CpG->Gene edges.
**Rationale**: The manufacturer's manifest provides the definitive mapping between CpG probe IDs and genomic coordinates / nearest genes. TSS proximity (<1500bp) is the standard definition for promoter-associated CpGs.
**Source**: Illumina support site, `humanmethylation450_15017482_v1-2.csv` (~192 MB)

---

## Cross-Cutting Decisions

### D-040: Team Structure -- Solo Engineer + Science Team
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Engineering work is solo (one person). Team provides science guidance, biology interpretation, and paper writing. Collaborating with a protein modeling team on ADC target discovery.
**Rationale**: Solo engineering with clear documentation (this decision log, development plan) is more efficient than coordinating multiple engineers on a research project with fast-changing requirements.
**Documented in**: `docs/Project_Scope_and_Pitch.md`

### D-041: Approach Pivot -- Foundation Models + Head (from Scratch GNN)
**Date**: Feb 2026 (Sessions 7-8)
**Decision**: Instead of training a GNN from scratch on 987 samples, use frozen foundation model embeddings as node features and train only a small GNN encoder + fusion head.
**Rationale**: 987 samples is too few to jointly learn biology representations AND pharmacology predictions (proven by negative R^2 on hard splits). Foundation models (scGPT: 51M params trained on millions of cells; CpGPT: 100M params trained on 150K+ samples) already encode rich biological knowledge. We only need to learn the drug response mapping on top.
**Analogy**: "Instead of teaching someone English from birth to translate a medical document, hire someone who already speaks English and just teach them medical terminology."
**Documented in**: `docs/Project_Scope_and_Pitch.md`, `docs/Refined_Strategy_Feb2026.md`

### D-042: Optimizer -- AdamW + Cosine Schedule
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use AdamW optimizer with cosine learning rate schedule for GNN training.
**Rationale**: Standard for transformer-adjacent architectures. Weight decay in AdamW prevents overfitting. Cosine schedule provides smooth annealing.
**Documented in**: `docs/Development_Plan.md` Section 7

### D-043: Mixed Precision -- BF16 on A100
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use BF16 (bfloat16) mixed precision for all GPU training.
**Rationale**: A100 GPUs natively support BF16 with no loss scaling needed (unlike FP16). Provides 2x throughput over FP32 with negligible accuracy impact.
**Documented in**: `docs/Refined_Strategy_Feb2026.md` Section 4.2

### D-044: Experiment Tracking -- W&B Offline Mode
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use Weights & Biases (W&B) in offline mode for experiment tracking on Delta HPC.
**Rationale**: Best HPC support (works without internet during jobs, syncs later). Free for academics. Rich visualization for comparing runs.
**Alternative considered**: MLflow (self-hosted, more setup), TensorBoard (less comparison support), Optuna dashboard only (insufficient for full tracking).

### D-045: Hyperparameter Search -- Optuna + SLURM Arrays
**Date**: Feb 17, 2026 (Session 8)
**Decision**: Use Optuna with SLURM job arrays for hyperparameter optimization. SQLite database on shared filesystem. One trial per array task.
**Rationale**: HPC-native approach. SLURM arrays parallelize trials across nodes. Optuna's TPE sampler is more efficient than grid search. SQLite avoids needing a database server.

---

## Success Criteria (defined Feb 17, 2026)

### Minimum Viable Paper
- Architecture B outperforms flat MLP baseline on at least one hard split
- Ablation shows graph structure contributes (random edges perform worse)
- Results reported on random + histology-blind + site-blind splits
- Comparison to TransCDR numbers (even if we don't beat cold-cell PCC 0.864)

### Strong Paper
- Architecture B beats TransCDR on cold-cell generalization
- Foundation model embeddings beat raw features (ablation #2)
- CpG-gene edges specifically contribute (ablation #7)
- Biological interpretation: top-attention edges correspond to known regulatory relationships
- Results on IMPROVE framework benchmarks

### Stretch Goals
- Cold-drug generalization (ECFP enables this)
- Cross-dataset validation (train GDSC, test CCLE)
- Attention visualization reveals novel CpG-gene regulatory candidates
- LoRA fine-tuning of scGPT beats frozen embeddings

### D-046: ECFP-4 Drug Encoding Validated and Complete
**Date**: Feb 26, 2026 (Session 8)
**Decision**: ECFP-4 fingerprints (2048-bit, Morgan radius=2) confirmed as drug encoding for the GNN pipeline. 314/375 modeled drugs encoded. 61 drugs without SMILES use one-hot fallback.
**Validation**: k=1 nearest-neighbor pathway concordance = 29.3% (4.0× above random chance of 7.3%). Monotonic similarity-concordance relationship confirmed. Mann-Whitney U p < 1e-16 for same- vs cross-pathway Tanimoto distributions. See `docs/Drug_Encoding_Report.md` for full results.
**Implementation**: `encode_drugs.py` produces `embeddings/drug_ecfp.npy`, `embeddings/drug_onehot.npy`, `embeddings/drug_index.csv`, and 9 validation figures.

### D-047: 2D Visualization (t-SNE/UMAP) Inappropriate for ECFP Validation
**Date**: Feb 26, 2026 (Session 8)
**Decision**: Replaced UMAP+HDBSCAN clustering figure with nearest-neighbor pathway concordance as primary validation. t-SNE retained as supplementary only.
**Rationale**: ECFP-4 produces 2048-bit sparse binary vectors where most drug pairs are nearly equidistant (Tanimoto ~0.1). Projecting N equidistant points to 2D is mathematically impossible without massive distortion (Wagen 2023). UMAP preserved only ~60% of true nearest-neighbor relationships. Nearest-neighbor concordance operates in the full 2048-dim space and is the standard cheminformatics validation (Riniker & Landrum 2013).
**Alternative considered**: UMAP+HDBSCAN was implemented and tested; found 82-86% noise regardless of parameter tuning. GTM (Generative Topographic Mapping) considered but not justified for this dataset size.

### D-048: Gene Expression Embedding — scGPT (OUTSTANDING)
**Date**: Feb 26, 2026 (Session 8) — planned, not yet implemented
**Decision**: Use scGPT pan-cancer checkpoint (51M params, 512-dim) for gene expression embeddings. Frozen inference, no fine-tuning on 987 samples.
**Status**: NOT YET STARTED. Requires: install scGPT on Delta, prepare CPM+log1p normalized input, run inference on 987 cell lines, save to `embeddings/scgpt_cell_embeddings.npy`.
**Fallback**: Geneformer V2 CLcancer, CancerFoundation, or PCA baseline.

### D-049: Methylation Embedding — CpGPT (OUTSTANDING)
**Date**: Feb 26, 2026 (Session 8) — planned, not yet implemented
**Decision**: Use CpGPT-100M (100M params, trained on 150K+ bulk array samples) for CpG methylation embeddings. Frozen inference.
**Status**: NOT YET STARTED. Requires: install CpGPT on Delta, format 10K CpG beta values, run inference on 987 cell lines, save to `embeddings/cpgpt_cpg_embeddings.npy`.
**Fallback**: MethylGPT (15M params, 256-dim) or PCA on top-N variable CpGs.

---

## Version History

| Date | Session | Entries Added |
|------|---------|---------------|
| 2026-02-26 | 8 | Initial creation, D-001 through D-045 covering all sessions 1-8 |
| 2026-02-26 | 8 (cont.) | D-046 through D-049: drug encoding completion, 2D viz limitation, outstanding FM embeddings |
