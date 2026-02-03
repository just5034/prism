# GNN Approaches for Drug Response Prediction

**Compiled**: January 7, 2026
**Project**: Pharmacoepigenomics - Drug Response Prediction
**Context**: Multi-omics data (DNA methylation + gene expression) from cancer cell lines

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Your Data Assets](#your-data-assets)
3. [Key GNN Architectures](#key-gnn-architectures)
4. [Multi-Omics Integration Strategies](#multi-omics-integration-strategies)
5. [Recommended Architecture](#recommended-architecture)
6. [Implementation Resources](#implementation-resources)
7. [Graph Construction Guide](#graph-construction-guide)
8. [Evaluation Strategy](#evaluation-strategy)
9. [Next Steps Checklist](#next-steps-checklist)
10. [References](#references)

---

## Executive Summary

Graph Neural Networks (GNNs) offer a promising approach for drug response prediction because they can encode **biological relationships** (gene regulation, protein interactions, pathway membership) that flat feature vectors miss. Recent research (2024-2025) shows GNNs outperforming traditional ML methods on GDSC drug response data, particularly for:

- **Generalization** to unseen cancer types
- **Interpretability** through attention weights and pathway-level analysis
- **Multi-omics integration** of methylation, expression, and mutation data

**Key insight**: Your "harder" generalization problem (predicting on unseen histologies/tissues) becomes more tractable if the model learns pathway-level patterns rather than memorizing individual feature correlations.

---

## Your Data Assets

### Current Datasets

| Dataset | Samples | Features | File |
|---------|---------|----------|------|
| **Multi-omics (PRIMARY)** | 987 cell lines | 29,265 (10K CpG + 19K genes) | `ml_with_gene_expr.csv.gz` |
| Methylation + Drug Response | 1,028 | 10,003 + IC50 | `ML_dataset_methylation_drug_response.csv.gz` |
| Methylation Only | 1,028 | 10,003 | `ML_dataset_methylation_features.csv.gz` |

### Feature Breakdown

- **DNA Methylation**: 10,000 most variable CpG sites (beta values 0-1)
- **Gene Expression**: 19,265 genes (RMA-normalized log2, range ~2-16)
- **Metadata**: Primary site (13 categories), histology (56 subtypes), COSMIC ID
- **Drug Response**: IC50 values from GDSC database

### Current Baselines

Your existing models (XGBoost, Random Forest) treat features independently. GNNs can potentially improve by capturing:
- Gene-gene regulatory relationships
- CpG-gene promoter associations
- Pathway co-membership patterns

---

## Key GNN Architectures

### 1. DRPreter (Interpretable, Pathway-Based)

**Paper**: [PMC9699175](https://pmc.ncbi.nlm.nih.gov/articles/PMC9699175/)

**Architecture**:
```
Cell Line:
  - Template graph: genes as nodes, PPI as edges
  - Divided into 34 KEGG pathway subgraphs
  - GAT processes each pathway separately
  - Concatenate pathway embeddings

Drug:
  - SMILES → molecular graph
  - GIN encoder for drug embedding

Fusion:
  - Type-aware transformer
  - Captures pathway-drug interactions
  - Residual connections preserve original features
```

**Strengths**:
- Pathway-level interpretability
- Knowledge-guided (uses KEGG)
- Good for mechanism discovery

**Relevance to your project**: Pathway patterns may transfer across cancer types, helping with your generalization challenge.

---

### 2. DeepCDR (Multi-Omics + Drug Graphs)

**Repository**: [github.com/kimmo1019/DeepCDR](https://github.com/kimmo1019/DeepCDR)

**Architecture**:
```
Cell Line Multi-Omics:
  - Mutation: 34,673 loci → neural network
  - Expression: 697 genes → neural network
  - Methylation: 808 sites → neural network
  - Concatenate embeddings

Drug:
  - Molecular graph (atoms as nodes, bonds as edges)
  - Node features: 75-dim (atom type, degree, hybridization)
  - Graph convolutional layers

Fusion:
  - Concatenate cell line + drug embeddings
  - Dense layers → IC50 prediction
```

**Strengths**:
- Already handles methylation + expression
- Proven on GDSC data
- Clean implementation available

**Relevance to your project**: Directly applicable architecture - you have the same data types.

---

### 3. NIHGCN (Heterogeneous Graph)

**Repository**: [github.com/weiba/NIHGCN](https://github.com/weiba/NIHGCN)

**Architecture**:
```
Heterogeneous Network:
  - Node types: drugs, cell lines
  - Edge types: known drug responses

Features:
  - Cell lines: gene expression (transformed)
  - Drugs: molecular fingerprints

Model:
  - Parallel GCN layers
  - Neighborhood interaction (NI) module
  - Predicts missing drug-cell line responses
```

**Strengths**:
- Tested on GDSC AND CCLE
- Includes 8 comparison algorithms
- Handles missing response values naturally

**Relevance to your project**: Benchmark-ready with your exact data source.

---

### 4. XGDP (Explainable, State-of-the-Art 2025)

**Paper**: [Nature Scientific Reports (Jan 2025)](https://www.nature.com/articles/s41598-024-83090-3)
**Repository**: [github.com/SCSE-Biomedical-Computing-Group/XGDP](https://github.com/SCSE-Biomedical-Computing-Group/XGDP)

**Architecture**:
```
Drug:
  - Molecular graph representation
  - GNN module for latent features

Cell Line:
  - Gene expression features
  - CNN compression

Fusion:
  - Two multi-head cross-attention layers
  - Drug-cell line feature interaction
```

**Strengths**:
- Latest published method (2025)
- Focus on explainability/mechanism prediction
- Cross-attention captures drug-gene relationships

**Relevance to your project**: State-of-the-art performance, interpretable predictions.

---

### 5. TGSA (Twin Graph, Similarity Augmentation)

**Repository**: [github.com/violet-sto/TGSA](https://github.com/violet-sto/TGSA)

**Architecture**:
```
Graph Construction:
  - Protein-protein association networks
  - Heterogeneous edges from similarity matrices

Twin GNN:
  - Parallel processing for drugs and cell lines
  - Similarity augmentation for cold-start

Output:
  - Drug response prediction
  - Handles new cell lines/drugs
```

**Strengths**:
- Addresses cold-start problem
- Similarity augmentation helps generalization
- Twin architecture for balanced learning

**Relevance to your project**: Designed for generalization to new samples - similar to your cross-cancer-type challenge.

---

### 6. GraphCDR (Contrastive Learning)

**Repository**: [github.com/BioMedicalBigDataMiningLab/GraphCDR](https://github.com/BioMedicalBigDataMiningLab/GraphCDR)

**Architecture**:
```
Contrastive Learning Framework:
  - Learn robust representations via contrastive objectives
  - Drug and cell line encoders
  - Prediction of missing IC50 values
```

**Strengths**:
- Contrastive learning improves generalization
- Handles sparse response matrices
- Self-supervised pre-training possible

---

### 7. GPDRP (Graph Transformer)

**Paper**: [BMC Bioinformatics](https://link.springer.com/article/10.1186/s12859-023-05618-0)

**Architecture**:
```
Drug:
  - Molecular graphs
  - Graph Transformer encoder

Cell Line:
  - Gene pathway activity scores
  - Deep neural network

Integration:
  - Multimodal fusion
  - Tested on bulk RNA-seq
```

**Strengths**:
- Graph Transformer (more expressive than GCN)
- Pathway activity as features (dimensionality reduction)
- Outperforms recent published models

---

## Multi-Omics Integration Strategies

### Strategy Comparison

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Multi-view GCN** | Separate graphs for methylation & expression, late fusion | Preserves omics-specific patterns | May miss cross-omics interactions |
| **Heterogeneous Graph** | Single graph with multiple node/edge types | Captures all relationships | Complex to implement |
| **Pathway-based** | Aggregate features per pathway, graph on pathways | Interpretable, reduces dimensions | Loses gene-level detail |
| **Bipartite Graph** | CpG sites ↔ genes with promoter edges | Mechanistic (methylation→expression) | Requires promoter annotations |
| **Concatenation** | Simple feature concatenation + GNN | Easy to implement | Doesn't model omics relationships |

### Recommended for Your Data

**Start with**: Pathway-based or Concatenation approach
- Pathway-based: Aggregate your 10K CpGs and 19K genes into ~300 KEGG pathways
- Concatenation: Gene nodes with [expression, mean_promoter_methylation] features

**Graduate to**: Heterogeneous graph if initial results promising
- Node types: genes, CpG sites, drugs
- Edge types: regulates, promoter_of, targets

---

## Recommended Architecture

Based on your data and the research survey, here's a concrete starting architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│              RECOMMENDED FIRST GNN ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GRAPH CONSTRUCTION (per cell line):                            │
│  ├─ Nodes: ~5,000 genes (most variable from your 19K)           │
│  ├─ Node features (per gene):                                    │
│  │   ├─ Gene expression value (1 dim)                           │
│  │   └─ Mean promoter methylation (1 dim)                       │
│  │   └─ [Optional] Variance, pathway membership (one-hot)       │
│  └─ Edges: STRING PPI (confidence > 700)                        │
│      └─ Alternative: KEGG pathway co-membership                 │
│                                                                  │
│  ENCODER:                                                        │
│  ├─ Input: Node features [N_genes × feature_dim]                │
│  ├─ Layer 1: GATConv(in_dim, 128, heads=4) → [N × 512]         │
│  ├─ Layer 2: GATConv(512, 128, heads=4) → [N × 512]            │
│  ├─ Layer 3: GATConv(512, 128, heads=1) → [N × 128]            │
│  ├─ Residual connections + LayerNorm after each                 │
│  └─ Readout: Attention pooling → [batch × 128]                  │
│                                                                  │
│  DRUG CONDITIONING:                                              │
│  ├─ Option A: Learned embeddings (num_drugs × 64)               │
│  └─ Option B: Drug molecular graph → GIN → embedding            │
│                                                                  │
│  PREDICTION HEAD:                                                │
│  ├─ Concatenate: [graph_embed (128) || drug_embed (64)]         │
│  ├─ MLP: Linear(192, 128) → ReLU → Dropout(0.3)                │
│  ├─ MLP: Linear(128, 64) → ReLU → Dropout(0.3)                 │
│  └─ Output: Linear(64, 1) → IC50                                │
│                                                                  │
│  TRAINING:                                                       │
│  ├─ Loss: MSE (masked for missing IC50 values)                  │
│  ├─ Optimizer: Adam, lr=1e-4, weight_decay=1e-5                 │
│  ├─ Batch size: 32-64 cell lines                                │
│  ├─ Validation: Your histology-based split                       │
│  └─ Early stopping: patience=20 on validation loss              │
│                                                                  │
│  FRAMEWORK:                                                      │
│  └─ PyTorch Geometric (pip install torch-geometric)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### PyTorch Geometric Skeleton

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader

class DrugResponseGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_drugs=265, dropout=0.3):
        super().__init__()

        # GAT encoder
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(hidden_dim * 4)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Drug embedding
        self.drug_embed = nn.Embedding(num_drugs, 64)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch, drug_idx):
        # GNN layers with residual connections
        h = self.conv1(x, edge_index)
        h = self.norm1(h).relu()

        h = self.conv2(h, edge_index)
        h = self.norm2(h).relu()

        h = self.conv3(h, edge_index)
        h = self.norm3(h)

        # Global pooling (graph-level embedding)
        h_graph = global_mean_pool(h, batch)  # [batch_size, hidden_dim]

        # Drug embedding
        h_drug = self.drug_embed(drug_idx)    # [batch_size, 64]

        # Predict IC50
        out = self.predictor(torch.cat([h_graph, h_drug], dim=-1))
        return out.squeeze(-1)


# Example data preparation
def create_cell_line_graph(expression, methylation, edge_index):
    """
    Create a PyG Data object for one cell line.

    Args:
        expression: [num_genes] gene expression values
        methylation: [num_genes] mean promoter methylation per gene
        edge_index: [2, num_edges] PPI edge indices

    Returns:
        PyG Data object
    """
    # Stack features: [num_genes, 2]
    x = torch.stack([expression, methylation], dim=-1)

    return Data(x=x, edge_index=edge_index)
```

---

## Implementation Resources

### GitHub Repositories

| Repository | Focus | Link |
|------------|-------|------|
| **DeepCDR** | Multi-omics + drug graphs | [github.com/kimmo1019/DeepCDR](https://github.com/kimmo1019/DeepCDR) |
| **NIHGCN** | Heterogeneous GCN, benchmarks | [github.com/weiba/NIHGCN](https://github.com/weiba/NIHGCN) |
| **TGSA** | Twin GNN, similarity augmentation | [github.com/violet-sto/TGSA](https://github.com/violet-sto/TGSA) |
| **GraphCDR** | Contrastive learning | [github.com/BioMedicalBigDataMiningLab/GraphCDR](https://github.com/BioMedicalBigDataMiningLab/GraphCDR) |
| **XGDP** | Explainable, 2025 SOTA | [github.com/SCSE-Biomedical-Computing-Group/XGDP](https://github.com/SCSE-Biomedical-Computing-Group/XGDP) |
| **DR_HGNN** | Heterogeneous GNN | [github.com/sshaghayeghs/DR_HGNN](https://github.com/sshaghayeghs/DR_HGNN) |

### Biological Network Databases

| Database | Content | URL |
|----------|---------|-----|
| **STRING** | Protein-protein interactions | [string-db.org](https://string-db.org) |
| **KEGG** | Pathway definitions | [kegg.jp](https://www.kegg.jp) |
| **Reactome** | Curated pathways | [reactome.org](https://reactome.org) |
| **TRRUST** | TF-target relationships | [grnpedia.org/trrust](https://www.grnpedia.org/trrust/) |
| **BioGRID** | Genetic interactions | [thebiogrid.org](https://thebiogrid.org) |

### Python Libraries

```bash
# Core GNN frameworks
pip install torch-geometric  # PyTorch Geometric (recommended)
pip install dgl              # Deep Graph Library (alternative)

# Biological data
pip install mygene           # Gene ID conversion
pip install bioservices      # Access to STRING, KEGG APIs

# Visualization
pip install networkx         # Graph manipulation
pip install pyvis            # Interactive graph visualization
```

---

## Graph Construction Guide

### Step 1: Get STRING PPI Network

```python
import pandas as pd
import requests

def download_string_ppi(species=9606, score_threshold=700):
    """
    Download STRING PPI network for human (9606).

    Args:
        species: NCBI taxonomy ID (9606 = human)
        score_threshold: Minimum combined score (0-1000)

    Returns:
        DataFrame with protein1, protein2, combined_score
    """
    url = f"https://stringdb-static.org/download/protein.links.v12.0/{species}.protein.links.v12.0.txt.gz"

    df = pd.read_csv(url, sep=' ', compression='gzip')
    df = df[df['combined_score'] >= score_threshold]

    # Convert ENSP IDs to gene symbols (requires mygene)
    # ... conversion code ...

    return df

# Alternative: Use pre-built edge lists from NIHGCN or DeepCDR repos
```

### Step 2: Map CpG Sites to Genes

```python
def map_cpg_to_genes(cpg_ids, annotation_file='illumina_450k_annotation.csv'):
    """
    Map CpG site IDs to nearest genes using Illumina annotation.

    For each gene, compute mean methylation of promoter CpGs
    (within 2kb of TSS).
    """
    # Load annotation (available from Illumina or GEO)
    annot = pd.read_csv(annotation_file)

    # Filter to promoter CpGs (TSS200, TSS1500, 5'UTR)
    promoter_cpgs = annot[annot['UCSC_RefGene_Group'].str.contains('TSS|5\'UTR', na=False)]

    # Group by gene, compute mean methylation
    # ... aggregation code ...

    return gene_methylation
```

### Step 3: Create PyG Dataset

```python
from torch_geometric.data import Dataset, Data
import torch

class CellLineDataset(Dataset):
    def __init__(self, expression_df, methylation_df, edge_index, drug_response_df):
        super().__init__()
        self.expression = expression_df
        self.methylation = methylation_df
        self.edge_index = edge_index
        self.drug_response = drug_response_df
        self.cell_lines = expression_df.index.tolist()
        self.drugs = drug_response_df.columns.tolist()

    def len(self):
        return len(self.cell_lines) * len(self.drugs)

    def get(self, idx):
        cell_idx = idx // len(self.drugs)
        drug_idx = idx % len(self.drugs)

        cell_line = self.cell_lines[cell_idx]
        drug = self.drugs[drug_idx]

        # Get features
        expr = torch.tensor(self.expression.loc[cell_line].values, dtype=torch.float)
        meth = torch.tensor(self.methylation.loc[cell_line].values, dtype=torch.float)
        x = torch.stack([expr, meth], dim=-1)

        # Get label
        y = self.drug_response.loc[cell_line, drug]

        return Data(
            x=x,
            edge_index=self.edge_index,
            drug_idx=torch.tensor(drug_idx),
            y=torch.tensor(y, dtype=torch.float) if not pd.isna(y) else None
        )
```

---

## Evaluation Strategy

### Baselines to Compare

1. **Your existing models**: XGBoost, Random Forest (already trained)
2. **MLP on same features**: Tests if graph structure helps vs just deep learning
3. **GCN variant**: Tests if attention (GAT) is worth the complexity

### Evaluation Splits

| Split | Description | Tests |
|-------|-------------|-------|
| **Random** | Standard 80/20 | Basic model performance |
| **Histology-based** | Hold out entire cancer subtypes | Generalization to unseen subtypes |
| **Site-based** | Hold out entire tissue types | Generalization to unseen tissues |
| **Drug-based** | Hold out specific drugs | Generalization to new drugs |

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | Mean squared error | Overall prediction accuracy |
| **R-squared** | 1 - MSE/Var(y) | Variance explained |
| **Pearson r** | Correlation(pred, true) | Rank preservation |
| **Per-drug R-squared** | R² computed per drug | Which drugs benefit most |

### Success Criteria

- **Minimum**: Match XGBoost baseline on random split
- **Target**: +5-10% R² improvement on histology/site splits
- **Stretch**: Interpretable attention weights identifying known drug targets

---

## Next Steps Checklist

### Phase 1: Data Preparation (1-2 days)
- [ ] Download STRING PPI network (human, score > 700)
- [ ] Map your 10K CpG sites to genes (promoter regions)
- [ ] Select ~5K most variable genes with both expression and methylation
- [ ] Create edge_index tensor from PPI network
- [ ] Build PyG Dataset class

### Phase 2: Baseline GNN (2-3 days)
- [ ] Implement 3-layer GAT encoder
- [ ] Add drug embedding layer
- [ ] Implement training loop with MSE loss
- [ ] Train on random split, validate
- [ ] Compare to XGBoost baseline

### Phase 3: Evaluation (1-2 days)
- [ ] Evaluate on histology-based split
- [ ] Evaluate on site-based split
- [ ] Compute per-drug R² scores
- [ ] Visualize attention weights

### Phase 4: Iteration (ongoing)
- [ ] Try GCN instead of GAT (simpler baseline)
- [ ] Try GIN (more expressive)
- [ ] Add pathway-level aggregation
- [ ] Experiment with drug molecular graphs (instead of embeddings)
- [ ] Consider heterogeneous graph approach

---

## References

### Primary Papers

1. **DRPreter** - Shin et al. (2022). "DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided Graph Neural Networks and Transformer." Int J Mol Sci. [PMC9699175](https://pmc.ncbi.nlm.nih.gov/articles/PMC9699175/)

2. **XGDP** - (2025). "Drug discovery and mechanism prediction with explainable graph neural networks." Scientific Reports. [Nature](https://www.nature.com/articles/s41598-024-83090-3)

3. **NIHGCN** - Wang et al. (2022). "Predicting cancer drug response using parallel heterogeneous graph convolutional networks with neighborhood interactions." Bioinformatics. [Oxford Academic](https://academic.oup.com/bioinformatics/article/38/19/4546/6673905)

4. **GPDRP** - (2023). "GPDRP: a multimodal framework for drug response prediction with graph transformer." BMC Bioinformatics. [Springer](https://link.springer.com/article/10.1186/s12859-023-05618-0)

5. **Multi-omics GNN Survey** - (2024). "A multimodal graph neural network framework for cancer molecular subtype classification." BMC Bioinformatics. [Springer](https://link.springer.com/article/10.1186/s12859-023-05622-4)

6. **Drug Synergy Review** - (2024). "A review on graph neural networks for predicting synergistic drug combinations." Artificial Intelligence Review. [Springer](https://link.springer.com/article/10.1007/s10462-023-10669-z)

### Related Resources

- Your existing guide: `GNN_Intuitive_Guide.md` (in project root)
- PyTorch Geometric documentation: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)
- Deep Graph Library: [dgl.ai](https://www.dgl.ai)

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-07 | Claude Code | Initial compilation from research session |

---

*This document serves as a reference for implementing GNN-based drug response prediction. Start with the recommended architecture and iterate based on your evaluation results.*
