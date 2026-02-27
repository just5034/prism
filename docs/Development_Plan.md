# Development Plan: PRISM Drug Response Prediction

**Created**: February 17, 2026
**Last Updated**: February 26, 2026
**Engineer**: Solo (team provides science guidance)
**Compute**: ~1000 A100 hours on NCSA Delta (NSF ACCESS)
**Related**: `docs/Refined_Strategy_Feb2026.md` (literature review + field assessment)

---

## 1. What We're Building

A model that predicts how cancer cell lines respond to drugs (IC50), using a novel graph that connects DNA methylation sites (CpGs) to genes. The graph captures real biology: CpGs regulate genes, genes interact in pathways. Foundation models provide rich node features. The model should generalize to cancer types it hasn't seen during training.

### One-Sentence Paper Pitch

> First heterogeneous GNN linking CpG methylation and gene expression via regulatory edges for drug response prediction, with foundation model node features, demonstrating improved generalization to unseen cancer types.

---

## 2. What We Have

### Data (in `data/processed/`)

| File | Contents | Shape |
|------|----------|-------|
| `ml_with_gene_expr.csv.gz` | CpG + gene expr + IC50 + metadata | 987 x 29,265 |
| `ML_dataset_methylation_drug_response.csv.gz` | CpG + IC50 + metadata | 925 x 10,375 |
| `ML_dataset_methylation_features.csv.gz` | CpG + metadata (no drugs) | 1,028 x 10,003 |

**Primary dataset**: `ml_with_gene_expr.csv.gz`
- 987 cell lines
- 10,000 CpG methylation features (top variable, beta values 0-1)
- 19,265 gene expression features (RMA-normalized log2)
- 375 drug IC50 values
- 3 metadata columns (primary site, histology, COSMIC ID)

### Code

| Path | Purpose |
|------|---------|
| `src/data/loading.py` | Data loading utilities (has known bug in `load_gse68379`) |
| `src/visualization/plot_gse270494.py` | Visualization pipeline |
| `model_selections.ipynb` | Single-drug baseline (Avagacestat, R² ~0.05-0.10) |
| `harder_ds.ipynb` | Hard split baselines (histology R² = -0.06, site R² = -0.02) |
| `multitask_drug_response.ipynb` | Multi-task framework across splits |
| `notebooks/reference/` | EDA and dataset prep notebooks (01-04) |

### Baseline Results (what we need to beat)

| Split | Best R² | Model | Notes |
|-------|---------|-------|-------|
| Random | 0.05-0.10 | Lasso / GAM | Single drug (Avagacestat) |
| Histology-blind | -0.06 | XGBoost | Worse than predicting mean |
| Site-blind | -0.02 | RF + PCA | Worse than predicting mean |

### External Benchmark (TransCDR, our primary competitor)

| Split | PCC | Notes |
|-------|-----|-------|
| Random (warm) | 0.936 | Multi-drug, methylation + expression + mutation |
| Cold cell line | 0.864 | Strongest published generalization with methylation |

---

## 3. Architecture

### 3.1 Overview

```
PHASE 1: EMBEDDING EXTRACTION (one-time, frozen models)
=========================================================

  Gene expression (987 x 19K)
       |
       v
    [scGPT, pan-cancer checkpoint, frozen]
       |
       v
    Per-gene or per-cell embeddings (512-dim)


  CpG methylation (987 x 10K)
       |
       v
    [CpGPT-100M, frozen]
       |
       v
    Per-CpG or per-sample embeddings


  Drug SMILES (375 drugs)
       |
       v
    [ECFP-4 fingerprints (2048-bit)]  -- always available, strong baseline
    [Pre-trained GIN (optional)]       -- for learned drug embedding (256-dim)


PHASE 2: GRAPH CONSTRUCTION (one-time, static)
=========================================================

  Nodes:
    - 10,000 CpG nodes
        features = CpGPT embeddings (or raw beta if CpGPT fails)
    - ~5,000 Gene nodes
        features = scGPT embeddings (or raw expression if scGPT fails)

  Edges (3 types, from external databases):
    - CpG --> Gene    : Illumina 450K manifest (TSS proximity, <1500bp)
    - Gene <-> Gene   : STRING v12 PPI (human 9606, confidence > 700)
    - CpG <-> CpG     : co-methylation or shared-gene annotation

  This graph is THE SAME for every cell line.
  What changes per cell line are the node FEATURES
  (each cell line has different expression / methylation values).


PHASE 3: MODEL (trained)
=========================================================

  Per cell line:
    - Populate node features from that cell line's data
    - Run HGT (Heterogeneous Graph Transformer), 2-3 layers
    - Mean-pool CpG nodes --> methylation representation
    - Mean-pool Gene nodes --> expression representation
    - Concat --> cell embedding

  Per (cell, drug) pair:
    - Cell embedding + Drug embedding
    - Fusion: bilinear attention or tensor product
    - Output: predicted IC50

  Training:
    - Multi-task: predict all 375 drugs simultaneously
    - Loss: MSE on IC50
    - Splits: random, histology-blind, site-blind
```

### 3.2 Why Each Component

| Component | Why |
|-----------|-----|
| CpG-Gene heterogeneous graph | Novel contribution. No one has done this for drug response. Encodes real regulatory biology. |
| Foundation model node features | CpGPT/scGPT embeddings are richer than raw scalars. Second novelty claim. |
| HGT encoder | Standard for heterogeneous graphs. Per-type projections handle CpG vs gene nodes naturally. |
| Multi-task IC50 prediction | 375 drugs share one model. Implicit regularization. Better than 375 separate models with 987 samples. |
| ECFP drug encoding | Proven competitive with neural encoders. Simple. One-hot is the fallback if we only test known drugs. |
| Hard split evaluation | This is where baselines fail and where our graph should shine. Required for publication credibility. |

### 3.3 Fallback: If the Graph Doesn't Help

If ablations show flat concatenation matches the GNN, the paper pivots to:
- "Foundation model embeddings (CpGPT + scGPT) for drug response" as the contribution
- Attention-based fusion instead of graph message passing
- Still publishable, just a different story

---

## 4. Implementation Phases

### Phase 0: Environment Setup (Week 1, Days 1-2)

**Where**: Delta HPC + Local

**Delta setup**:
```
# Packages to install
pytorch >= 2.1 (with CUDA)
torch-geometric (PyG)
scgpt
CpGPT
helical            # unified FM API for quick benchmarking
rdkit              # for ECFP fingerprints
optuna             # hyperparameter optimization
wandb              # experiment tracking (offline mode on Delta)
```

**Local setup**:
```
# Same packages minus GPU-heavy ones
# Plus: jupyter, matplotlib, seaborn for analysis
```

**Deliverables**:
- [x] Delta environment working with GPU access confirmed
- [ ] scGPT imports and loads checkpoint successfully ← **OUTSTANDING**
- [ ] CpGPT imports and loads checkpoint successfully ← **OUTSTANDING**
- [ ] PyG imports and basic graph operations work ← **OUTSTANDING**
- [x] Can load `ml_with_gene_expr.csv.gz` on Delta
- [x] RDKit installed and working on Delta

**Decision gate**: If CpGPT won't install cleanly, fall back to MethylGPT or PCA.

---

### Phase 1: Embedding Extraction (Week 1, Days 3-5)

**Where**: Delta (~5-15 A100 hours)
**Goal**: Pre-compute all foundation model embeddings. Save to disk. Never run these models again.

**Status**: PARTIALLY COMPLETE — drug encoding done, gene expression and methylation embeddings outstanding.

**Tasks**:

1. **Prepare input data**
   - Gene expression: CPM + log1p normalization of the 19K gene features
   - Map gene names to scGPT vocabulary (handle missing genes)
   - CpG methylation: format 10K CpG beta values for CpGPT input
   - ~~Drug SMILES: collect from GDSC metadata or PubChem for 375 drugs~~ ✅ DONE (314/375 with SMILES)

2. **Run scGPT inference** ← **OUTSTANDING**
   - Load pan-cancer checkpoint
   - Feed 987 cell line expression profiles
   - Extract 512-dim cell embeddings (and per-gene embeddings if available)
   - Save: `embeddings/scgpt_cell_embeddings.npy` (987 x 512)
   - Save: `embeddings/scgpt_gene_embeddings.npy` (987 x N_genes x 512) if feasible

3. **Run CpGPT inference** ← **OUTSTANDING**
   - Load CpGPT-100M
   - Feed 987 cell line methylation profiles
   - Extract per-sample embeddings (and per-CpG embeddings if available)
   - Save: `embeddings/cpgpt_sample_embeddings.npy`
   - Save: `embeddings/cpgpt_cpg_embeddings.npy` if feasible

4. **Run alternative expression models (via Helical)** ← **OUTSTANDING**
   - Geneformer V2 CLcancer checkpoint
   - CancerFoundation (smaller, cancer-specific)
   - Save each to `embeddings/` for comparison in Phase 2

5. ~~**Extract drug features**~~ ✅ DONE (Feb 26, 2026)
   - ✅ ECFP-4 fingerprints (2048-bit) for 314/375 drugs using RDKit
   - ✅ One-hot drug encoding (375-dim) as baseline
   - ✅ Saved: `embeddings/drug_ecfp.npy` (314, 2048), `embeddings/drug_onehot.npy` (375, 375)
   - ✅ Saved: `embeddings/drug_index.csv` with metadata mapping
   - ✅ Validated: 9 figures including k-NN concordance (4.0× enrichment at k=1)
   - ✅ Documented: `docs/Drug_Encoding_Report.md`

**Deliverables**:
- [x] `embeddings/drug_ecfp.npy` — ECFP-4 fingerprints (314, 2048)
- [x] `embeddings/drug_onehot.npy` — one-hot baseline (375, 375)
- [x] `embeddings/drug_index.csv` — drug name ↔ index mapping
- [ ] `embeddings/scgpt_cell_embeddings.npy` — gene expression embeddings ← **OUTSTANDING**
- [ ] `embeddings/scgpt_gene_embeddings.npy` — per-gene embeddings ← **OUTSTANDING**
- [ ] `embeddings/cpgpt_sample_embeddings.npy` — methylation sample embeddings ← **OUTSTANDING**
- [ ] `embeddings/cpgpt_cpg_embeddings.npy` — per-CpG embeddings ← **OUTSTANDING**
- [ ] Alternative FM embeddings (Geneformer, CancerFoundation) ← **OUTSTANDING**
- [ ] Validation: embedding shapes, no NaNs, reasonable value ranges
- [ ] Total storage: ~50-200 MB (trivially small)

**Decision gate**: Check embedding quality. Do cancer types cluster in PCA of scGPT embeddings? If not, the embeddings may not capture useful biology.

---

### Phase 2: Tabular Baselines (Week 2)

**Where**: Local (CPU), zero GPU hours
**Goal**: Establish strong baselines on foundation model embeddings. These may already beat prior work.

**Tasks**:

1. **Data preparation**
   - Concatenate scGPT (512) + CpGPT (N) embeddings per cell line
   - Create train/test splits: random, histology-blind, site-blind
   - For multi-task: target matrix is 987 x 375 (IC50 values, sparse)

2. **Single-task baselines (per-drug models)**
   - XGBoost, Random Forest, Ridge Regression on embeddings → IC50
   - Evaluate on top-20 most-tested drugs first
   - Compare: embeddings vs raw PCA features vs current baselines

3. **Multi-task baselines**
   - MLP: [concat_embedding; drug_encoding] → IC50
   - Architecture: 2-3 hidden layers, 256-128-64, dropout 0.3
   - Train on all (cell, drug) pairs jointly
   - Drug encoding: try both one-hot and ECFP

4. **Evaluate on all three splits**
   - Random split (5-fold CV)
   - Histology-blind split (leave-one-histology-out)
   - Site-blind split (leave-one-site-out)
   - Metrics: PCC, R², RMSE

5. **Compare foundation model variants**
   - scGPT vs Geneformer V2 vs CancerFoundation (expression side)
   - CpGPT vs MethylGPT vs PCA (methylation side)
   - Pick best combination for downstream GNN work

**Deliverables**:
- [ ] Results table: baselines x splits x metrics
- [ ] Best foundation model pair identified
- [ ] Comparison to prior baselines (Session 7 results)
- [ ] If embeddings already beat TransCDR on cold-cell → paper is viable even without GNN

**Decision gate**: If foundation model embeddings + simple MLP already generalize well, the GNN may add marginal value. Proceed with GNN anyway (it's the novelty), but adjust expectations.

---

### Phase 3: Graph Construction (Week 3)

**Where**: Local (CPU)
**Goal**: Build the CpG-Gene heterogeneous graph.

**Tasks**:

1. **Download external data**
   - STRING v12: `9606.protein.links.v12.0.txt.gz` (human PPI)
   - Illumina 450K manifest: `HumanMethylation450_15017482_v1-2.csv`
     (maps each CpG probe to nearest gene, distance to TSS)
   - Gene name mapping: Ensembl ↔ HGNC symbol ↔ scGPT vocabulary

2. **Build CpG → Gene edges**
   - From 450K manifest: each CpG maps to 0-N nearby genes (within 1500bp of TSS)
   - Filter to our 10K CpG sites and genes present in our expression data
   - Expected: ~15K-30K edges

3. **Build Gene ↔ Gene edges**
   - From STRING PPI: filter to confidence > 700
   - Filter to genes in our dataset (~5K genes)
   - Map STRING protein IDs → gene symbols
   - Expected: ~50K-200K edges

4. **Build CpG ↔ CpG edges (optional)**
   - Option A: CpGs mapping to the same gene
   - Option B: Co-methylation (Pearson > 0.8 across cell lines)
   - Start without these, add in ablation

5. **Create PyG HeteroData object**
   - Node types: `cpg` (10K), `gene` (~5K)
   - Edge types: `(cpg, regulates, gene)`, `(gene, interacts, gene)`, optionally `(cpg, co_methyl, cpg)`
   - Node features: placeholder (filled per-sample at training time)
   - Save graph structure: `data/graphs/cpg_gene_hetgraph.pt`

6. **Validate graph**
   - Check connectivity: is the graph one component or fragmented?
   - Check degree distributions: any isolated nodes?
   - Visualize a subgraph (e.g., TP53 neighborhood)

**Deliverables**:
- [ ] `data/graphs/cpg_gene_hetgraph.pt` - static graph structure
- [ ] `data/graphs/gene_name_mapping.csv` - ID mapping table
- [ ] Graph statistics: nodes, edges per type, avg degree, components
- [ ] Validation: known biology checks (e.g., BRCA1 connected to DNA repair genes?)

**Decision gate**: If >30% of CpG nodes have zero edges to genes, the 450K manifest coverage may be insufficient. Consider expanding TSS distance threshold or adding genomic-distance-based edges.

---

### Phase 4: Architecture A - Baseline GNN (Weeks 3-4)

**Where**: Delta (~50-100 A100 hours)
**Goal**: Get a working GNN training pipeline. Establish graph-based baseline.

Architecture A is simpler: GAT on the gene-gene PPI subgraph only (no CpG nodes yet). This validates the infrastructure before adding heterogeneous complexity.

**Tasks**:

1. **Implement data pipeline**
   - `src/data/dataset.py`: PyG Dataset class
   - Per-sample: populate gene node features from expression + embeddings
   - Batch construction: same graph structure, different node features per sample
   - Train/val/test split logic (random + hard splits)

2. **Implement Architecture A**
   - `src/models/arch_a_gat.py`
   - 2-3 GAT layers on gene-gene PPI graph
   - Node features: scGPT gene embeddings (512-dim) for each cell line
   - Graph readout: mean pool → cell embedding
   - Fusion: concat cell embedding + drug ECFP → MLP → IC50
   - Multi-task: shared encoder, drug-conditioned output

3. **Implement training loop**
   - `src/models/training.py`
   - MSE loss on IC50
   - Adam optimizer, learning rate scheduling
   - Early stopping on validation loss
   - BF16 mixed precision
   - W&B logging (offline mode)
   - Checkpoint saving

4. **Implement evaluation**
   - `src/models/evaluation.py`
   - Metrics: PCC, Spearman, R², RMSE per drug and overall
   - Split evaluation: random, histology-blind, site-blind
   - Comparison to Phase 2 tabular baselines

5. **Train and evaluate**
   - Random split first (sanity check: should improve over MLP baseline)
   - Then hard splits
   - Basic hyperparameter tuning (learning rate, hidden dim, num layers)

**Deliverables**:
- [ ] Working training pipeline (reusable for Architecture B)
- [ ] Architecture A results on all splits
- [ ] Confirmed: GNN infrastructure works on Delta

**Decision gate**: If Architecture A doesn't beat the MLP baseline on random splits, debug before proceeding. Likely issues: learning rate too high, not enough epochs, data loading bug.

---

### Phase 5: Architecture B - CpG-Gene HetGraph (Weeks 5-7)

**Where**: Delta (~200-400 A100 hours)
**Goal**: Implement and optimize the novel contribution.

**Tasks**:

1. **Implement HGT encoder**
   - `src/models/arch_b_hetgraph.py`
   - Heterogeneous Graph Transformer (HGT) from PyG
   - Per-node-type linear projections (CpG dim → hidden, Gene dim → hidden)
   - Multi-head attention across edge types
   - 2-3 HGT layers, 128 hidden dim, 4-8 attention heads

2. **Per-sample feature population**
   - CpG nodes: CpGPT embedding concatenated with raw beta value
   - Gene nodes: scGPT embedding concatenated with raw expression value
   - Handle: node features change per cell line, graph structure stays fixed

3. **Dual readout**
   - Mean pool CpG nodes → methylation representation
   - Mean pool Gene nodes → expression representation
   - Concat → cell embedding

4. **Drug fusion**
   - Option A: Bilinear attention (cell_emb, drug_emb → IC50)
   - Option B: Tensor product
   - Option C: Simple concat + MLP (baseline fusion)
   - Compare all three

5. **Hyperparameter sweep**
   - Optuna + SLURM job arrays, ~200 trials
   - Search space: hidden_dim {64, 128, 256}, num_layers {2, 3, 4},
     heads {4, 8}, lr {1e-4, 5e-4, 1e-3}, dropout {0.1, 0.3, 0.5},
     fusion_type {bilinear, tensor, concat}

6. **Evaluate on all splits**
   - 5-fold CV on random split
   - Leave-one-histology-out (the key test)
   - Leave-one-site-out
   - Compare to Architecture A, MLP baseline, and TransCDR numbers

**Deliverables**:
- [ ] Architecture B implementation
- [ ] Best hyperparameters from sweep
- [ ] Results table: Arch B vs Arch A vs MLP vs literature
- [ ] Performance on hard splits (the paper's main result)

**Decision gate**: If Arch B doesn't beat Arch A on hard splits, check if CpG-gene edges are contributing (ablation). If not, the graph structure may not be adding value beyond the foundation model embeddings.

---

### Phase 6: Ablations + Paper Results (Week 8+)

**Where**: Delta (~100-200 A100 hours) + Local (analysis)
**Goal**: Prove each component matters. Generate publication-ready results.

**Ablation matrix** (each row = one experiment, 3-5 seeds each):

| # | Experiment | What It Proves |
|---|-----------|----------------|
| 1 | Full model (Arch B) | Reference performance |
| 2 | Raw features instead of FM embeddings | Value of foundation models |
| 3 | Gene-only graph (remove CpG nodes) | Value of methylation graph structure |
| 4 | CpG-only graph (remove gene nodes) | Value of expression graph structure |
| 5 | No graph (flat concat of all embeddings → MLP) | Value of graph structure itself |
| 6 | Random edges (same density as real) | Real biology vs just regularization |
| 7 | Remove CpG→Gene edges only | Value of regulatory connections |
| 8 | Remove Gene↔Gene edges only | Value of PPI network |
| 9 | One-hot drug encoding instead of ECFP | Value of molecular structure |
| 10 | Single-task (per-drug) instead of multi-task | Value of multi-task learning |

**Paper figures to generate**:
- Performance bar chart: all methods x all splits
- Ablation table (the above)
- Graph visualization: subgraph around a known drug target
- Attention heatmap: which edges get highest attention weight
- t-SNE/UMAP of learned cell embeddings, colored by cancer type
- Per-drug performance scatter: our method vs best baseline
- Generalization gap plot: random-split performance vs hard-split performance

**Deliverables**:
- [ ] Complete ablation table with error bars
- [ ] All paper figures generated
- [ ] Results section draft
- [ ] Supplementary data prepared

---

## 5. File Structure (Target)

```
prism/
├── src/
│   ├── data/
│   │   ├── loading.py          # existing data loading
│   │   ├── dataset.py          # NEW: PyG dataset class
│   │   ├── preprocessing.py    # NEW: FM input formatting
│   │   └── graph_builder.py    # NEW: build CpG-gene graph
│   ├── models/
│   │   ├── arch_a_gat.py       # NEW: Architecture A
│   │   ├── arch_b_hetgraph.py  # NEW: Architecture B (primary)
│   │   ├── baselines.py        # NEW: MLP, XGBoost wrappers
│   │   ├── drug_encoder.py     # NEW: ECFP + optional GIN
│   │   ├── fusion.py           # NEW: bilinear, tensor product
│   │   ├── training.py         # NEW: training loop
│   │   └── evaluation.py       # NEW: metrics + split logic
│   ├── embeddings/
│   │   └── extract.py          # NEW: FM inference scripts
│   └── visualization/
│       ├── plot_gse270494.py   # existing
│       └── paper_figures.py    # NEW: publication figures
├── configs/
│   ├── arch_a.yaml             # NEW: Architecture A config
│   ├── arch_b.yaml             # NEW: Architecture B config
│   └── sweep.yaml              # NEW: Optuna search space
├── scripts/
│   ├── extract_embeddings.sh   # NEW: SLURM script for Phase 1
│   ├── train.sh                # NEW: SLURM training script
│   └── sweep.sh                # NEW: SLURM array sweep script
├── embeddings/                  # NEW: pre-computed FM outputs
├── data/
│   ├── processed/              # existing ML datasets
│   └── graphs/                 # NEW: PyG graph files
├── results/                     # NEW: experiment outputs
├── docs/
│   ├── Development_Plan.md     # THIS FILE
│   ├── Refined_Strategy_Feb2026.md
│   └── GNN_Architecture_Proposals.md
└── notebooks/
    └── reference/              # existing EDA notebooks
```

---

## 6. External Data to Download

| Data | Source | Size | Used For |
|------|--------|------|----------|
| STRING v12 PPI | string-db.org (9606.protein.links.v12.0.txt.gz) | ~200 MB | Gene↔Gene edges |
| Illumina 450K manifest | Illumina support site or bioconductor | ~200 MB | CpG→Gene edges |
| Drug SMILES | GDSC metadata or PubChem (by drug name) | <1 MB | ECFP fingerprints |
| scGPT checkpoint | HuggingFace (tdc/scGPT) or pip | ~200 MB | Gene expression embeddings |
| CpGPT checkpoint | pip install CpGPT (downloads from AWS) | ~1 GB | Methylation embeddings |

---

## 7. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Foundation model freezing | Frozen (no fine-tuning) | 987 samples too few. Overfitting risk. Budget saved for GNN. |
| Drug encoding | ECFP-4 (primary) + one-hot (ablation) | Neural drug encoders don't beat ECFP. Cold-drug not our focus. |
| Multi-task vs single-task | Multi-task | 375 drugs x 987 samples. Shared encoder provides regularization. |
| GNN architecture | HGT (PyG built-in) | Standard for heterogeneous graphs. Well-tested in PyG. |
| Fusion | Bilinear attention (primary) | Captures multiplicative cell-drug interactions. Compare to concat in ablation. |
| Split strategy | Random + histology-blind + site-blind | Matches prior work. Hard splits are the key result. |
| Optimizer | AdamW + cosine schedule | Standard for transformer-adjacent architectures. |
| Mixed precision | BF16 | A100 native. 2x throughput. No loss scaling needed. |
| Experiment tracking | W&B offline mode | Best HPC support. Free for academics. |
| Hyperparameter search | Optuna + SLURM arrays | Simple, effective, HPC-native. |
| Random seed | 42 | Project convention. Multiple seeds for final results. |

---

## 8. Success Criteria

### Minimum Viable Paper
- [ ] Architecture B outperforms flat MLP baseline on at least one hard split
- [ ] Ablation shows graph structure contributes (random edges perform worse)
- [ ] Results reported on random + histology-blind + site-blind splits
- [ ] Comparison to TransCDR numbers (even if we don't beat cold-cell PCC 0.864)

### Strong Paper
- [ ] Architecture B beats TransCDR on cold-cell generalization
- [ ] Foundation model embeddings beat raw features (ablation #2)
- [ ] CpG-gene edges specifically contribute (ablation #7)
- [ ] Biological interpretation: top-attention edges correspond to known regulatory relationships
- [ ] Results on IMPROVE framework benchmarks (GDSC1 + GDSC2 + CTRPv2)

### Stretch Goals
- [ ] Cold-drug generalization (ECFP enables this)
- [ ] Cross-dataset validation (train GDSC, test CCLE)
- [ ] Attention visualization reveals novel CpG-gene regulatory candidates
- [ ] LoRA fine-tuning of scGPT beats frozen embeddings

---

## 9. Risk Mitigation Checklist

| Week | Check | If It Fails |
|------|-------|-------------|
| 1 | CpGPT installs and produces embeddings | Use MethylGPT or PCA |
| 1 | scGPT works on bulk cell line profiles | Use CancerFoundation or Geneformer V2 CLcancer |
| 2 | FM embeddings beat PCA baseline on random split | Graph structure must carry the weight; adjust expectations |
| 3 | >70% of CpG nodes have gene edges | Expand TSS distance threshold (1500bp → 5000bp) |
| 3 | STRING PPI maps to our gene set | Try BioGRID as alternative PPI source |
| 4 | Architecture A beats MLP on random split | Debug data pipeline before proceeding |
| 5-7 | Architecture B beats Architecture A | Check if CpG nodes contribute via ablation #3 |
| 5-7 | Hard split R² > 0 | The minimum bar. If negative, revisit feature engineering |

---

## 10. Timeline Summary

| Week | Phase | Key Deliverable | GPU Hours |
|------|-------|----------------|-----------|
| 1 | 0 + 1 | Environment + all embeddings extracted | 5-15 |
| 2 | 2 | Tabular baseline results table | 0 (CPU) |
| 3 | 3 + 4 start | Graph built + Arch A pipeline working | 20-50 |
| 4 | 4 | Arch A results on all splits | 30-50 |
| 5 | 5 start | Arch B HGT implemented + first training runs | 50-100 |
| 6 | 5 | Arch B hyperparameter sweep | 100-200 |
| 7 | 5 | Arch B final results on all splits | 50-100 |
| 8 | 6 | Ablation study + paper figures | 100-200 |
| **Total** | | | **355-715** |
| **Buffer** | | Debugging, reruns, additional experiments | **285-645** |
