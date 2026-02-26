# Refined Research Strategy - February 2026

**Date**: February 17, 2026
**Context**: Solo engineer, ~1000 A100 hours on Delta, team is science-oriented
**Goal**: Publish a novel cancer drug response prediction method using CpG-gene graph structure

---

## 1. Field Assessment: What Changed Since Session 7

### 1.1 Your Novelty Gap Still Exists

**No published method models CpG-gene regulatory relationships as graph structure for drug response prediction.** Confirmed as of Feb 2026 across all major venues.

- GraphAge (PNAS Nexus 2025) proved CpG graphs work biologically, but for age prediction only
- GraphMeXplain (bioRxiv Jan 2026) uses CpG graphs for methylation analysis, not drug response
- All drug response GNNs (GraphTCDR, HGACL-DRP, DRPreter) use gene-only or drug-only graphs
- DeepCDR and TransCDR use methylation as flat vectors, not graph-structured

**This is your paper's core claim.** Guard it.

### 1.2 New Competitors to Benchmark Against

| Method | Year | PCC (Random) | PCC (Cold Cell) | Uses Methylation | Uses Graphs |
|--------|------|-------------|-----------------|------------------|-------------|
| TransCDR | 2024 | 0.936 | 0.864 | Yes (flat) | GIN (drug only) |
| DeepCCDS | 2025 | 0.93 | -- | No | No |
| GraphTCDR | 2025 | SOTA on PRISM | -- | Yes (flat) | HetGNN (cell-drug) |
| HGACL-DRP | 2025 | AUC 0.99 | -- | Yes | HetGraph + contrastive |
| scGPT+DeepCDR | 2025 | > DeepCDR | -- | No | GCN (drug only) |

**Your target to beat**: TransCDR's cold-cell PCC of 0.864. This is the strongest generalization result from a method that also uses methylation.

### 1.3 Foundation Models Are Now Viable

Two critical new models appeared since your last review:

- **CpGPT** (bioRxiv Oct 2024): First true methylation foundation model. 100M params, trained on 150K+ samples, supports 450K arrays natively. MIT license.
- **MethylGPT** (bioRxiv Oct 2024): 15M params, trained on 154K methylomes, 256-dim CLS embedding. Apache 2.0.

Neither has been applied to drug response. Using their embeddings as CpG node features in your graph is a second novelty claim.

### 1.4 Key Benchmark Finding

The **IMPROVE framework** (BIB 2025, NCI-DOE) is now the standard evaluation protocol. Key finding: **CTRPv2 is the most effective source dataset for training** cross-dataset models. Consider training on CTRPv2 in addition to GDSC.

The **scDrugMap** study (Nature Communications 2025) benchmarked 8 foundation models for drug response: scGPT won zero-shot, UCE won after fine-tuning, scFoundation won pooled evaluation.

### 1.5 Drug Encoder Update

A rigorous benchmark of 25 pretrained molecular models (arXiv Aug 2025) found: **nearly all neural models show negligible improvement over ECFP fingerprints.** Top performers: CLAMP > R-MAT > MolBERT > ChemBERTa-MTR. GNNs ranked worst. This means: don't over-invest in fancy drug encoders. ECFP + a simple GIN is fine.

---

## 2. Foundation Model Selections

### 2.1 Gene Expression Encoder: scGPT

| Property | Value |
|----------|-------|
| Model | scGPT (pan-cancer checkpoint) |
| Parameters | 51M |
| Output | 512-dim cell embedding |
| Install | `pip install scgpt` |
| License | MIT |
| Validation | Directly validated for IC50 prediction via DeepCDR integration (arXiv Apr 2025) |
| Input format | CPM + log1p normalized expression, zero-pad missing genes |

**Why scGPT over Geneformer**: scGPT won zero-shot evaluation in scDrugMap, has simpler input format (no rank-value encoding), has direct drug response validation, and is smaller (51M vs 316M for Geneformer V2).

**Fallback**: Geneformer V2 with CLcancer checkpoint (continual-learned on 14M cancer transcriptomes). Apache 2.0 on HuggingFace.

**Risk**: Both are single-cell models being applied to bulk cell line data. Mitigate by always comparing against PCA baseline on raw features.

### 2.2 DNA Methylation Encoder: CpGPT

| Property | Value |
|----------|-------|
| Model | CpGPT-100M |
| Parameters | 100M |
| Output | Sample-level embedding (dimension TBD from repo) |
| Install | `pip install CpGPT` |
| License | MIT |
| GitHub | github.com/lcamillo/CpGPT |
| Trained on | 150K+ samples, 2000+ datasets, 450K/EPIC arrays |

**Why CpGPT**: Trained on bulk array data (exact domain match for your 450K data), not single-cell. Supports 450K arrays natively.

**Fallback**: MethylGPT (15M params, 256-dim CLS embedding, faster, Apache 2.0). Or PCA on top-N variable CpGs (strong baseline per Genome Biology 2025).

### 2.3 Drug Encoder: GIN + ECFP Dual

| Component | Purpose |
|-----------|---------|
| ECFP-4 fingerprints (2048-bit) | Strong baseline, no GPU needed, proven competitive |
| Pre-trained GIN (from DeepCDR/TransCDR) | Graph-level drug embedding for PyG integration |

Don't invest in Uni-Mol2 or ChemBERTa unless you need a camera-ready boost. ECFP alone matches or beats most neural encoders.

### 2.4 Rapid Prototyping Tool: Helical

`pip install helical` provides a unified API to extract embeddings from scGPT, Geneformer, UCE, and others through a single interface. Use this to benchmark multiple expression encoders in a day.

---

## 3. Revised Architecture Plan

### 3.1 The Hybrid: Foundation Embeddings + Novel CpG-Gene Graph

Your old Architecture B (CpG-gene heterogeneous graph) was designed to operate on raw features. The refined version uses **foundation model embeddings as node features**, combining the best of both approaches:

```
                    FOUNDATION MODELS (frozen, run once)
                    ====================================
Gene expression --> scGPT ---------> 512-dim per-gene embeddings
                                     (or 512-dim cell embedding)

CpG methylation --> CpGPT ---------> per-CpG embeddings
                                     (or sample-level embedding)

Drug SMILES -----> GIN + ECFP -----> 256-dim drug embedding


                    YOUR NOVEL CONTRIBUTION
                    ====================================
                    CpG-Gene Heterogeneous Graph

                    Nodes:
                      - 10K CpG nodes (features = CpGPT embeddings)
                      - 5K Gene nodes (features = scGPT embeddings)

                    Edges (3 types):
                      - CpG --> Gene (Illumina 450K manifest, TSS proximity)
                      - Gene --> Gene (STRING PPI, confidence > 700)
                      - CpG --> CpG (co-methylation, same gene)

                    Encoder: HGT (Heterogeneous Graph Transformer)
                      - 2-3 layers, 128 hidden dim
                      - Per-node-type linear projections
                      - Multi-head attention across edge types

                    Readout:
                      - Mean pool CpG nodes --> CpG representation
                      - Mean pool Gene nodes --> Gene representation
                      - Concat [CpG_repr; Gene_repr] --> Cell embedding

                    Fusion with drug:
                      - Tensor product or bilinear attention
                      - [Cell_embedding x Drug_embedding] --> IC50
```

### 3.2 Why This Architecture

1. **Novel**: First CpG-gene heterogeneous graph for drug response. Primary paper claim.
2. **Foundation-powered**: CpGPT/scGPT embeddings as node features. Second novelty claim.
3. **Biologically grounded**: Edges encode real regulatory relationships (CpG silences gene, genes interact in pathways).
4. **Generalization-oriented**: Graph structure captures tissue-agnostic biology (pathway-level patterns transfer across cancer types).
5. **Compute-feasible**: Graph is static (built once), 15K nodes is small for PyG, training on 987 samples is fast.

### 3.3 Ablation Study Design

To prove each component matters, run these ablations:

| Ablation | What It Tests |
|----------|---------------|
| Raw features instead of foundation embeddings | Value of pre-training |
| Gene-only graph (no CpG nodes) | Value of CpG graph structure |
| CpG-only graph (no gene nodes) | Value of gene expression |
| Flat concatenation (no graph) | Value of graph structure itself |
| Random edges (same density) | Whether real biology matters vs just regularization |
| Remove CpG-Gene edges | Value of regulatory connections |
| Remove Gene-Gene edges | Value of PPI network |
| MLP head instead of GNN | Is the graph better than tabular? |

---

## 4. Compute Budget (1000 A100 Hours)

### 4.1 Phase Allocation

| Phase | A100 Hours | What You Get |
|-------|-----------|-------------|
| **Phase 0**: Embedding extraction | 5-15 | All 987 samples through scGPT + CpGPT + GIN. Pre-computed embeddings saved to disk (~50 MB total). Everything downstream operates on these. |
| **Phase 1**: Tabular baselines on embeddings | 0 (CPU) | MLP, XGBoost, RF on concatenated embeddings. Hundreds of configs. Run on your laptop. Publishable baselines. |
| **Phase 2**: Architecture A (GAT on STRING PPI) | 50-100 | Baseline GNN. Validates infrastructure, establishes graph-based baseline. |
| **Phase 3**: Architecture B (CpG-Gene HetGraph) | 200-400 | **Primary contribution.** Full hyperparameter sweep + ablation study. |
| **Phase 4**: Cross-validation + hard splits | 100-200 | 5-fold CV on random, histology-blind, site-blind splits. Paper-ready numbers. |
| **Phase 5**: Ablation studies | 100-200 | 8 ablations x multiple seeds. Proves each component matters. |
| **Buffer** | 85-545 | Debugging, reruns, additional experiments |

### 4.2 Key Compute Savings

1. **Pre-compute embeddings once** (Phase 0). Total output is ~50 MB. Everything trains on this.
2. **All tabular baselines on CPU** (Phase 1). XGBoost on 987 x 1024-dim takes seconds.
3. **BF16 mixed precision** on A100 for all GPU work. 2x throughput.
4. **Single GPU per job**. 987 samples does not warrant multi-GPU.
5. **Optuna + SLURM job arrays** for sweeps. One trial per array task, SQLite on shared FS.
6. **Do NOT fine-tune foundation models** unless frozen embeddings clearly fail. 987 samples = overfitting risk.

### 4.3 What Can Run Locally (Free)

- All Phase 1 tabular experiments (CPU)
- Graph construction (building adjacency matrices from STRING, 450K manifest)
- Data preprocessing, feature engineering
- Visualization, analysis, paper writing
- Small MLP head training (consumer GPU or CPU)

---

## 5. Implementation Timeline

### Week 1: Foundation + Embeddings

**Goal**: Extract all foundation model embeddings. Establish baseline performance.

| Day | Task | Where |
|-----|------|-------|
| 1 | Install scGPT, CpGPT, helical on Delta. Test imports. | Delta |
| 2 | Prepare input data: CPM+log1p expression, 450K methylation, drug SMILES. | Local |
| 3 | Run scGPT inference on 987 cell lines. Save 512-dim embeddings. | Delta (~2-3 hrs) |
| 4 | Run CpGPT inference on 987 cell lines. Save methylation embeddings. | Delta (~2-3 hrs) |
| 5 | Extract ECFP fingerprints + GIN drug embeddings for 375 drugs. | Local / Delta |

**Deliverable**: `embeddings/` folder with pre-computed vectors for all samples and drugs.

### Week 2: Tabular Baselines (mostly local)

**Goal**: Publishable baseline numbers using foundation model embeddings.

| Day | Task | Where |
|-----|------|-------|
| 1-2 | Train MLP, XGBoost, RF on scGPT+CpGPT concatenated embeddings → IC50. Multi-task (all 375 drugs). | Local (CPU) |
| 3 | Evaluate on random split, histology-blind split, site-blind split. | Local |
| 4 | Compare: foundation embeddings vs raw PCA features vs your current baselines. | Local |
| 5 | Use Helical to swap in Geneformer V2 cancer-tuned. Compare. | Delta (~2 hrs) |

**Deliverable**: Baseline results table. Decision on which foundation model pair works best.

### Week 3-4: Graph Construction + Architecture A

**Goal**: Build CpG-gene graph. Implement and validate baseline GNN (Architecture A).

| Task | Details |
|------|---------|
| Download STRING v12 PPI | Human (9606), confidence > 700, ~500K edges |
| Download Illumina 450K manifest | CpG-to-gene annotations (TSS proximity) |
| Build heterogeneous graph | 10K CpG + 5K gene nodes, 3 edge types |
| Implement Architecture A | GAT on STRING PPI subgraph + drug GIN |
| Train + evaluate | Random split + cold-cell split |

**Deliverable**: Working GNN training pipeline. Architecture A results.

### Week 5-7: Architecture B (Primary Contribution)

**Goal**: Implement and optimize the novel CpG-Gene heterogeneous graph.

| Task | Details |
|------|---------|
| Implement HGT encoder | Per-node-type projections, multi-head attention |
| Foundation embeddings as node features | CpGPT for CpG nodes, scGPT for gene nodes |
| Tensor product fusion | Cell embedding x drug embedding |
| Hyperparameter sweep | Optuna, ~200 trials on Delta |
| Hard split evaluation | Histology-blind, site-blind, leave-one-cancer-type-out |

**Deliverable**: Architecture B results across all splits. Comparison vs Architecture A and tabular baselines.

### Week 8: Ablations + Paper Prep

**Goal**: Prove each component matters. Generate paper figures.

| Task | Details |
|------|---------|
| Run 8 ablations | See table in Section 3.3 |
| Multiple seeds | 3-5 random seeds per configuration |
| Generate figures | PCA of learned embeddings, attention heatmaps, performance bar charts |
| Draft results section | Tables and figures for paper |

**Deliverable**: Complete experimental results. Ablation table. Ready for paper writing.

---

## 6. Novelty Claims for the Paper

### Claim 1 (Primary)
First heterogeneous graph neural network encoding CpG-gene regulatory relationships for cancer drug response prediction.

### Claim 2
First application of DNA methylation foundation model (CpGPT) embeddings as node features in a drug response GNN.

### Claim 3
Demonstration that graph-structured epigenetic-transcriptomic relationships improve generalization to unseen cancer types (leave-one-type-out), where flat-vector methods fail (negative R-squared).

### Paper Framing
> "We present the first heterogeneous graph neural network that jointly models CpG methylation sites and gene expression nodes with regulatory edges for cancer drug response prediction. By encoding epigenetic-transcriptomic regulatory relationships as graph structure and leveraging foundation model embeddings (scGPT, CpGPT) as node features, our model captures tissue-agnostic drug sensitivity patterns that generalize across unseen cancer types."

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| scGPT/CpGPT embeddings don't improve over PCA | Medium | Medium | This is why ablations exist. If raw PCA is competitive, the graph structure is still novel. |
| Bulk-to-single-cell domain gap (scGPT) | Medium | Medium | CpGPT is trained on bulk data. For expression, always compare against Geneformer V2 cancer checkpoint. |
| CpGPT installation issues / poor docs | Medium | Low | Fallback to MethylGPT or PCA. |
| Graph structure doesn't help (flat concat equals GNN) | Low-Medium | High | Random-edges ablation will diagnose this. If edges don't matter, pivot to attention-based fusion. |
| Compute overrun | Low | Medium | 545-hour buffer. Tabular baselines require zero GPU. |
| Someone publishes CpG-gene graph for DRP first | Low | Critical | Move fast. Week 1 starts NOW. |

---

## 8. Immediate Action Items (This Week)

1. **Clone repo to Delta** (if not already done)
2. **Install**: PyTorch 2.x, PyG, scGPT, CpGPT, helical, optuna, wandb
3. **Prepare input data**: Format expression + methylation for foundation model input
4. **Run embedding extraction** (scGPT + CpGPT on 987 samples)
5. **Download STRING v12** and **Illumina 450K manifest**
6. **Train first tabular baseline** on concatenated embeddings

---

## 9. Key References

### Foundation Models
- scGPT: Cui et al., Nature Methods 2024
- Geneformer V2: Theodoris et al., Nature 2023 + V2 update
- CpGPT: Camillo et al., bioRxiv Oct 2024 (github.com/lcamillo/CpGPT)
- MethylGPT: bioRxiv Oct 2024
- Helical framework: github.com/helicalAI/helical

### Drug Response Prediction
- TransCDR: Xia et al., BMC Biology 2024 (github.com/XiaoqiongXia/TransCDR)
- DeepCCDS: Han et al., Advanced Science 2025
- GraphTCDR: Neural Networks Aug 2025
- HGACL-DRP: medRxiv Sep 2025
- CRISP: Nature Computational Science 2025
- IMPROVE framework: BIB 2025 (NCI-DOE standardized benchmarking)

### Drug Encoders
- Molecular embedding benchmark: arXiv:2508.06199 (Aug 2025)
- DeepCDR GIN: Bioinformatics 2020

### CpG Graph Precedents
- GraphAge: PNAS Nexus 2025 (CpG co-methylation graph for age)
- GraphMeXplain: bioRxiv Jan 2026 (CpG graphs, not drug response)

### Benchmarks
- scDrugMap: Nature Communications 2025 (8 foundation models for DRP)
- scGPT+DeepCDR: arXiv:2504.14361 (Apr 2025)
