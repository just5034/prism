# GNN Architecture Proposals for Cancer Drug Response Prediction

**Date**: February 7, 2026
**Project**: Pharmacoepigenomics - Multi-Omics Drug Response Prediction
**Target Environment**: NCSA Delta (A100 GPUs)
**Framework**: PyTorch Geometric (PyG)

---

## Table of Contents

1. [Problem Statement & Data Context](#1-problem-statement--data-context)
2. [Baseline Performance & Motivation](#2-baseline-performance--motivation)
3. [Literature Context (2025-2026)](#3-literature-context-2025-2026)
4. [Architecture A: GAT on STRING PPI](#4-architecture-a-gat-on-string-ppi)
5. [Architecture B: CpG-Gene Heterogeneous Graph](#5-architecture-b-cpg-gene-heterogeneous-graph)
6. [Architecture C: Pathway-Hierarchical Graph Transformer + Contrastive Pre-training](#6-architecture-c-pathway-hierarchical-graph-transformer--contrastive-pre-training)
7. [Shared Components](#7-shared-components)
8. [Architecture Comparison](#8-architecture-comparison)
9. [Recommended Progression](#9-recommended-progression)
10. [References](#10-references)

---

## 1. Problem Statement & Data Context

### Task

Predict cancer cell line drug sensitivity (IC50) from multi-omics molecular profiles using graph-structured biological knowledge.

### Data

| Asset | Description | Shape |
|-------|-------------|-------|
| **Primary dataset** | `ml_with_gene_expr.csv.gz` | 987 cell lines x 29,265 features |
| **Methylation features** | Top 10K most variable CpG sites (beta values 0-1) | 987 x 10,000 |
| **Gene expression** | RMA-normalized log2 expression | 987 x 19,265 genes |
| **Drug response** | IC50 values from GDSC | 987 x 375 drugs |
| **Metadata** | Primary site (13), histology (56), COSMIC ID | 987 x 3 |

### Key Challenge

Models must generalize across cancer types. Current baselines fail on hard splits:

| Split Strategy | Best Baseline R² | Model |
|---------------|-------------------|-------|
| Random | 0.05-0.10 | Lasso / GAM |
| Histology-based | -0.06 | XGBoost |
| Site-based | -0.02 | Random Forest + PCA |

The negative R² on generalization splits means flat-vector models predict worse than predicting the mean. Graph structure encoding biological relationships (pathways, regulation) may capture transferable patterns that persist across cancer types.

### External Data Sources Needed

| Source | Content | Used In |
|--------|---------|---------|
| **STRING v12** | Protein-protein interactions (human, 9606) | All architectures |
| **Illumina 450K manifest** | CpG-to-gene annotations (TSS proximity) | Architecture B |
| **MSigDB / KEGG / Reactome** | Pathway gene sets | Architecture C |
| **PubChem / ChEMBL** | Drug SMILES strings | All architectures (drug encoder) |
| **GDSC** | Drug metadata + IC50 matrix | All architectures |

---

## 2. Baseline Performance & Motivation

### Why Flat-Vector Models Fail on Hard Splits

1. **Memorization over generalization**: Random Forest / XGBoost learn correlations specific to cancer types present in training (e.g., "high methylation at cg12345 + lung tissue = resistant"). These correlations break when the test set contains unseen cancer types.

2. **No biological structure**: A flat vector treats TP53 expression and a random CpG site as equally unrelated. In reality, TP53 pathway genes co-vary and respond to drugs as a unit.

3. **Curse of dimensionality**: 29K features vs 987 samples. PCA helps but discards the biologically meaningful structure.

### Why GNNs May Help

- **Pathway-level patterns transfer**: "DNA repair pathway active + PARP inhibitor = sensitive" generalizes across cancer types.
- **Regularization through structure**: Graph connectivity constrains what the model can learn - genes must influence each other through known biological relationships.
- **Multi-scale readout**: Learn gene-level, pathway-level, and cell-level representations simultaneously.

---

## 3. Literature Context (2025-2026)

### Critical Gap Identified

**No published method models CpG-gene regulatory relationships as graph structure for drug response prediction.** This is the primary novelty opportunity.

- GraphMeXplain (bioRxiv Jan 2026) and GraphAge (PNAS Nexus 2025) demonstrate that CpG-based graphs work for biological prediction, but neither targets drug response.
- All existing drug response GNNs (DRPreter, DeepCDR, TGSA, GraphTCDR, etc.) use gene-only or drug-only graphs.

### Key Methods Informing Our Designs

| Method | Key Innovation | What We Borrow |
|--------|---------------|----------------|
| **GPS** (NeurIPS 2022) | Local MPNN + global attention | Architecture C cell encoder |
| **TransCDR** (BMC Bio 2024) | Triple drug encoder + methylation branch | Multi-encoder drug design |
| **Tensor Product Fusion** (BIB 2024) | Multiplicative cell-drug interaction | Fusion layer (all architectures) |
| **DRPreter** (IJMS 2022) | Gene-to-pathway hierarchy with GAT | Architecture C hierarchical design |
| **RT-DMF** (BIB 2025) | Denoising + imputation of IC50 matrix | Missing data preprocessing |
| **GraphTCDR** (Neural Networks 2025) | Cell-drug heterogeneous graph | Architecture B heterogeneous design |
| **PASO** (PLOS Comp Bio 2025) | Pathway-aware SMILES-omics attention | Architecture C drug-pathway attention |
| **BANDRP** (BIB 2024) | Multi-head bilinear drug-pathway attention | Architecture C fusion layer |

---

## 4. Architecture A: GAT on STRING PPI

### 4.1 Overview

**Objective**: Strong, reproducible baseline that establishes whether graph structure helps at all. Builds on well-tested patterns from DRPreter / DeepCDR.

**Novelty level**: Engineering contribution (not publishable on architecture alone, but provides the foundation for Architectures B and C).

**Timeline**: 2-3 weeks.

### 4.2 Architecture Diagram

```
                    DRUG BRANCH                          CELL LINE BRANCH

    SMILES ──→ [Pre-trained GIN] ──→ h_drug [256]      Gene Expression (5K genes)
                                          │                    │
                                          │              Map CpG → Gene
                                          │              (mean promoter meth)
                                          │                    │
                                          │              Node features: [expr, meth]
                                          │                    │
                                          │              STRING PPI Graph (5K nodes)
                                          │                    │
                                          │              ┌─────────────────┐
                                          │              │  GAT Layer 1    │
                                          │              │  (4 heads, 128) │
                                          │              ├─────────────────┤
                                          │              │  GAT Layer 2    │
                                          │              │  (4 heads, 128) │
                                          │              ├─────────────────┤
                                          │              │  GAT Layer 3    │
                                          │              │  (1 head, 256)  │
                                          │              └────────┬────────┘
                                          │                       │
                                          │              Attention Pooling
                                          │                       │
                                          │                  h_cell [256]
                                          │                       │
                                          ▼                       ▼
                                    ┌─────────────────────────────────┐
                                    │   Tensor Product Partial Fusion │
                                    │   h_cell ⊗ h_drug → h_fused    │
                                    └───────────────┬─────────────────┘
                                                    │
                                              MLP Head [256→128→1]
                                                    │
                                                  IC50
```

### 4.3 Graph Construction

**Nodes**: ~5,000 genes (intersection of expression features and STRING proteins)

**Node features per cell line** (2 dimensions per gene):
- `gene_expression`: RMA-normalized log2 value (standardized across cell lines)
- `mean_promoter_methylation`: Average beta value of CpG sites mapped to this gene's promoter region (TSS200, TSS1500, 5'UTR from Illumina 450K manifest)

**Edges**: STRING PPI v12, human (taxon 9606), combined_score >= 700

**Expected graph statistics**:
- ~5,000 nodes
- ~50,000-100,000 edges
- Average degree: ~20-40
- Small enough for full-batch training on a single GPU

```python
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def build_string_ppi_graph(gene_list, score_threshold=700):
    """
    Build STRING PPI edge_index for a given set of genes.

    Parameters
    ----------
    gene_list : list of str
        Gene symbols present in our expression data.
    score_threshold : int
        Minimum STRING combined score (0-1000).

    Returns
    -------
    edge_index : torch.LongTensor [2, num_edges]
    gene_to_idx : dict mapping gene symbol -> node index
    """
    # Load STRING links file (pre-downloaded)
    # Format: protein1 protein2 combined_score
    string_df = pd.read_csv(
        'data/external/9606.protein.links.v12.0.txt.gz',
        sep=' ', compression='gzip'
    )
    string_df = string_df[string_df['combined_score'] >= score_threshold]

    # Load STRING aliases to map ENSP IDs -> gene symbols
    aliases = pd.read_csv(
        'data/external/9606.protein.aliases.v12.0.txt.gz',
        sep='\t', compression='gzip'
    )
    # Filter to gene symbol source (BioMart_HUGO)
    hugo = aliases[aliases['source'] == 'BioMart_HUGO']
    ensp_to_gene = dict(zip(hugo['#string_protein_id'], hugo['alias']))

    # Map to gene symbols
    string_df['gene1'] = string_df['protein1'].map(ensp_to_gene)
    string_df['gene2'] = string_df['protein2'].map(ensp_to_gene)
    string_df = string_df.dropna(subset=['gene1', 'gene2'])

    # Filter to genes in our dataset
    gene_set = set(gene_list)
    mask = string_df['gene1'].isin(gene_set) & string_df['gene2'].isin(gene_set)
    string_df = string_df[mask]

    # Build index mapping
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Build edge_index (undirected: add both directions)
    src = string_df['gene1'].map(gene_to_idx).values
    dst = string_df['gene2'].map(gene_to_idx).values
    edge_index = torch.tensor(
        np.stack([np.concatenate([src, dst]),
                  np.concatenate([dst, src])]),
        dtype=torch.long
    )

    return edge_index, gene_to_idx


def map_cpg_to_genes(cpg_ids, gene_list):
    """
    Map CpG sites to genes using Illumina 450K manifest.
    Returns mean promoter methylation per gene.

    Parameters
    ----------
    cpg_ids : list of str
        CpG probe IDs (e.g., 'cg00944421').
    gene_list : list of str
        Target gene symbols.

    Returns
    -------
    cpg_gene_map : dict of {gene: [cpg_ids]}
    """
    # Load Illumina 450K manifest (pre-downloaded)
    manifest = pd.read_csv(
        'data/external/HumanMethylation450_15017482_v1-2.csv',
        skiprows=7, low_memory=False,
        usecols=['IlmnID', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group']
    )

    # Filter to promoter regions
    promoter_mask = manifest['UCSC_RefGene_Group'].str.contains(
        'TSS200|TSS1500|5\'UTR', na=False
    )
    promoter = manifest[promoter_mask]

    # Filter to our CpG sites
    promoter = promoter[promoter['IlmnID'].isin(cpg_ids)]

    # Explode multi-gene annotations (some CpGs map to multiple genes)
    promoter = promoter.assign(
        gene=promoter['UCSC_RefGene_Name'].str.split(';')
    ).explode('gene')

    # Filter to our gene list
    gene_set = set(gene_list)
    promoter = promoter[promoter['gene'].isin(gene_set)]

    # Group CpGs by gene
    cpg_gene_map = promoter.groupby('gene')['IlmnID'].apply(list).to_dict()

    return cpg_gene_map


def create_cell_line_graph(cell_line_row, gene_list, edge_index,
                           cpg_gene_map, expr_cols, meth_cols):
    """
    Create a PyG Data object for one cell line.

    Parameters
    ----------
    cell_line_row : pd.Series
        One row from the multi-omics dataset.
    gene_list : list of str
        Ordered list of gene symbols (defines node ordering).
    edge_index : torch.LongTensor [2, E]
    cpg_gene_map : dict of {gene: [cpg_ids]}
    expr_cols : list of str
        Expression column names (format: 'expr_GENE').
    meth_cols : list of str
        Methylation column names (format: 'cgXXXXXXXX').

    Returns
    -------
    data : torch_geometric.data.Data
    """
    num_genes = len(gene_list)

    # Expression features
    expr_values = np.array([
        cell_line_row.get(f'expr_{gene}', 0.0) for gene in gene_list
    ], dtype=np.float32)

    # Mean promoter methylation per gene
    meth_values = np.zeros(num_genes, dtype=np.float32)
    for i, gene in enumerate(gene_list):
        cpgs = cpg_gene_map.get(gene, [])
        if cpgs:
            vals = [cell_line_row.get(cpg, np.nan) for cpg in cpgs]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                meth_values[i] = np.mean(vals)

    # Stack features: [num_genes, 2]
    x = torch.tensor(
        np.stack([expr_values, meth_values], axis=-1),
        dtype=torch.float
    )

    return Data(x=x, edge_index=edge_index)
```

### 4.4 Drug Encoder

Use a pre-trained Graph Isomorphism Network (GIN) from MoleculeNet. This avoids training a drug encoder from scratch on our limited data.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from rdkit import Chem
from torch_geometric.utils import from_smiles

class PretrainedDrugEncoder(nn.Module):
    """
    GIN encoder for drug molecular graphs.

    Pre-train on MoleculeNet datasets (BBBP, HIV, etc.) or load
    weights from an existing checkpoint. For initial experiments,
    can also use fixed Morgan fingerprints as a simpler alternative.
    """
    def __init__(self, num_atom_features=9, hidden_dim=128,
                 out_dim=256, num_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = num_atom_features if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.projection = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = x.relu()

        # Graph-level readout
        h = global_mean_pool(x, batch)
        return self.projection(h)


def precompute_drug_embeddings(smiles_list, encoder, device='cuda'):
    """
    Pre-compute drug embeddings for all 375 compounds.
    Run once, save, and load during training.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings for each drug.
    encoder : PretrainedDrugEncoder
        Pre-trained or frozen drug encoder.

    Returns
    -------
    embeddings : torch.Tensor [num_drugs, out_dim]
    """
    encoder.eval()
    embeddings = []

    with torch.no_grad():
        for smiles in smiles_list:
            data = from_smiles(smiles).to(device)
            batch = torch.zeros(data.num_nodes, dtype=torch.long,
                              device=device)
            h = encoder(data.x.float(), data.edge_index, batch)
            embeddings.append(h.squeeze(0))

    return torch.stack(embeddings)
```

**Simpler alternative for initial experiments**: Use pre-computed Morgan fingerprints (2048-bit) passed through an MLP instead of a GIN encoder. This removes the dependency on drug molecular graphs and lets you focus on the cell line side.

```python
class FingerprintDrugEncoder(nn.Module):
    """Morgan fingerprint -> dense embedding."""
    def __init__(self, fp_dim=2048, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )

    def forward(self, fp):
        return self.mlp(fp)
```

### 4.5 Model Architecture

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class AttentionPooling(nn.Module):
    """Learned attention pooling over graph nodes."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, batch):
        # x: [total_nodes, hidden_dim]
        # batch: [total_nodes] mapping nodes to graphs
        attn_weights = self.attn(x)  # [total_nodes, 1]

        # Softmax within each graph
        from torch_geometric.utils import softmax
        attn_weights = softmax(attn_weights.squeeze(-1), batch)

        # Weighted sum per graph
        x_weighted = x * attn_weights.unsqueeze(-1)
        from torch_geometric.nn import global_add_pool
        return global_add_pool(x_weighted, batch)


class TensorProductFusion(nn.Module):
    """
    Low-rank bilinear (tensor product partial) fusion.
    Captures multiplicative interactions between cell and drug.
    Outperforms concatenation per BIB 2024 systematic comparison.
    """
    def __init__(self, cell_dim, drug_dim, rank=128, out_dim=256):
        super().__init__()
        self.proj_cell = nn.Linear(cell_dim, rank)
        self.proj_drug = nn.Linear(drug_dim, rank)
        self.output = nn.Sequential(
            nn.LayerNorm(rank),
            nn.ReLU(),
            nn.Linear(rank, out_dim)
        )

    def forward(self, h_cell, h_drug):
        # Project to shared rank space
        h_c = self.proj_cell(h_cell)   # [batch, rank]
        h_d = self.proj_drug(h_drug)   # [batch, rank]
        # Element-wise product (Hadamard) = low-rank bilinear
        h_fused = h_c * h_d            # [batch, rank]
        return self.output(h_fused)


class ArchitectureA(nn.Module):
    """
    Architecture A: GAT on STRING PPI with pre-trained drug encoder.

    Cell line: 3-layer GAT on gene PPI graph -> attention pooling -> h_cell
    Drug: Pre-trained GIN (or fingerprint MLP) -> h_drug
    Fusion: Tensor product partial -> MLP -> IC50
    """
    def __init__(self, gene_feat_dim=2, hidden_dim=128,
                 drug_dim=256, num_heads=4, dropout=0.3):
        super().__init__()

        # --- Cell line encoder (GAT) ---
        self.conv1 = GATConv(gene_feat_dim, hidden_dim, heads=num_heads,
                             concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim,
                             heads=num_heads, concat=True, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim * 2,
                             heads=1, concat=False, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm3 = nn.LayerNorm(hidden_dim * 2)

        self.pool = AttentionPooling(hidden_dim * 2)

        # --- Fusion ---
        self.fusion = TensorProductFusion(
            cell_dim=hidden_dim * 2,  # 256
            drug_dim=drug_dim,         # 256
            rank=128,
            out_dim=256
        )

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, h_drug):
        """
        Parameters
        ----------
        x : [total_nodes, gene_feat_dim]
        edge_index : [2, total_edges]
        batch : [total_nodes]
        h_drug : [batch_size, drug_dim] pre-computed drug embeddings
        """
        # GAT layers with residual connections
        h = self.conv1(x, edge_index)
        h = self.norm1(h).relu()
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.norm2(h).relu()
        h = self.dropout(h)

        h = self.conv3(h, edge_index)
        h = self.norm3(h)

        # Attention pooling -> graph-level embedding
        h_cell = self.pool(h, batch)  # [batch_size, 256]

        # Fusion
        h_fused = self.fusion(h_cell, h_drug)  # [batch_size, 256]

        # Predict IC50
        return self.head(h_fused).squeeze(-1)
```

### 4.6 Training Strategy

```python
import torch
from torch_geometric.loader import DataLoader

def train_architecture_a():
    """End-to-end training pipeline for Architecture A."""

    # --- Hyperparameters ---
    config = {
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'epochs': 200,
        'patience': 30,
        'hidden_dim': 128,
        'num_heads': 4,
        'dropout': 0.3,
        'seed': 42,
    }

    # --- Data loading ---
    # 1. Build shared graph structure (same for all cell lines)
    edge_index, gene_to_idx = build_string_ppi_graph(gene_list)

    # 2. Pre-compute drug embeddings (or load fingerprints)
    drug_embeddings = precompute_drug_embeddings(smiles_list, drug_encoder)
    # drug_embeddings: [375, 256]

    # 3. Create per-(cell_line, drug) training samples
    # Each sample = (cell_line_graph, drug_idx, ic50)
    # Filter to non-missing IC50 entries only
    dataset = DrugResponseDataset(
        omics_df=df,
        gene_list=gene_list,
        edge_index=edge_index,
        cpg_gene_map=cpg_gene_map,
        drug_response_df=drug_response_df,
        drug_embeddings=drug_embeddings
    )

    # 4. Split (histology-based for hard evaluation)
    train_dataset, val_dataset, test_dataset = histology_split(dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # --- Model ---
    model = ArchitectureA(
        gene_feat_dim=2,
        hidden_dim=config['hidden_dim'],
        drug_dim=256,
        num_heads=config['num_heads'],
        dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    # --- Training loop ---
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(
                batch.x, batch.edge_index, batch.batch,
                drug_embeddings[batch.drug_idx].to(device)
            )

            # MSE loss on observed IC50 values
            loss = nn.functional.mse_loss(pred, batch.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        # Validation
        val_loss, val_r2 = evaluate(model, val_loader, drug_embeddings)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/arch_a_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break

    # Final evaluation
    model.load_state_dict(torch.load('models/arch_a_best.pt'))
    test_loss, test_r2 = evaluate(model, test_loader, drug_embeddings)
    return test_r2
```

### 4.7 Implementation Plan

| Step | Task | Deliverable | Est. Time |
|------|------|-------------|-----------|
| A.1 | Download STRING v12 PPI + aliases for human | `data/external/9606.protein.links.v12.0.txt.gz` | 1 day |
| A.2 | Download Illumina 450K manifest | `data/external/HumanMethylation450_manifest.csv` | 1 day |
| A.3 | Download drug SMILES from PubChem/GDSC | `data/external/drug_smiles.csv` | 1 day |
| A.4 | Write graph construction pipeline | `src/graphs/string_ppi.py` | 2 days |
| A.5 | Write CpG-to-gene mapping | `src/graphs/cpg_gene_map.py` | 1 day |
| A.6 | Write PyG Dataset class | `src/data/graph_dataset.py` | 2 days |
| A.7 | Implement drug encoder (fingerprint MLP first, GIN later) | `src/models/drug_encoder.py` | 1 day |
| A.8 | Implement GAT cell line encoder + fusion + head | `src/models/architecture_a.py` | 2 days |
| A.9 | Implement training loop + evaluation | `src/training/train.py` | 2 days |
| A.10 | Implement split strategies (random, histology, site) | `src/data/splits.py` | 1 day |
| A.11 | Run experiments on Delta, tune hyperparameters | `results/arch_a/` | 3 days |
| **Total** | | | **~16 days** |

### 4.8 Expected Outcomes

- **Random split**: R² 0.15-0.25 (modest improvement over baselines, since random split already somewhat works)
- **Histology split**: R² 0.02-0.08 (the real test -- any positive R² is a win over current -0.06)
- **Site split**: R² 0.01-0.06 (positive R² is a win over current -0.02)
- **Interpretability**: GAT attention weights identify which gene-gene interactions matter per drug

---

## 5. Architecture B: CpG-Gene Heterogeneous Graph

### 5.1 Overview

**Objective**: Model the epigenetic regulatory relationship between CpG methylation and gene expression as explicit graph structure. This is the primary novelty contribution.

**Novelty claim**: *"First GNN to model CpG-gene regulatory relationships as graph structure for drug response prediction."*

**Timeline**: 4-6 weeks.

### 5.2 Architecture Diagram

```
                DRUG BRANCH                        CELL LINE BRANCH

    SMILES → [GIN Encoder] → h_drug [256]    ┌──────────────────────────┐
                    │                         │  HETEROGENEOUS GRAPH      │
                    │                         │                          │
                    │                         │  CpG nodes (10K)         │
                    │                         │    │ features: beta val  │
                    │                         │    │                     │
                    │                         │    ├─ regulates ──→ Gene │
                    │                         │    │  (from 450K         │
                    │                         │    │   manifest)         │
                    │                         │                          │
                    │                         │  Gene nodes (5K)         │
                    │                         │    │ features: expr val  │
                    │                         │    │                     │
                    │                         │    ├─ interacts ──→ Gene │
                    │                         │    │  (STRING PPI)       │
                    │                         │                          │
                    │                         │  CpG ←─ co-meth ─→ CpG  │
                    │                         │    (optional, top-k      │
                    │                         │     correlated pairs)    │
                    │                         └──────────┬───────────────┘
                    │                                    │
                    │                         ┌──────────▼───────────────┐
                    │                         │  HeteroGNN Layers (x3)   │
                    │                         │  (per-relation message   │
                    │                         │   passing with HGTConv   │
                    │                         │   or HeteroConv)         │
                    │                         └──────────┬───────────────┘
                    │                                    │
                    │                         Dual-stream Pooling:
                    │                         ├─ Gene pool → h_gene [128]
                    │                         └─ CpG pool  → h_cpg  [128]
                    │                                    │
                    │                              Concat → h_cell [256]
                    │                                    │
                    ▼                                    ▼
              ┌─────────────────────────────────────────────┐
              │       Tensor Product Partial Fusion          │
              │       h_cell ⊗ h_drug → h_fused [256]       │
              └──────────────────┬──────────────────────────┘
                                 │
                           MLP Head → IC50
```

### 5.3 Heterogeneous Graph Construction

The key innovation is the three-relation heterogeneous graph:

**Node types**:
1. **Gene nodes** (5K): Feature = standardized gene expression (1-dim per cell line)
2. **CpG nodes** (10K): Feature = beta methylation value (1-dim per cell line)

**Edge types**:
1. **CpG → Gene** ("regulates"): From Illumina 450K manifest, CpG in promoter region (TSS200, TSS1500, 5'UTR) of gene. ~20K-40K edges.
2. **Gene ↔ Gene** ("interacts"): From STRING PPI, combined_score >= 700. ~50K-100K edges.
3. **CpG ↔ CpG** ("co-methylated"): Top-k correlated CpG pairs (|Pearson r| > 0.7 across 987 cell lines). Optional, adds ~50K edges.

**Total graph**: ~15K nodes, ~120K-190K edges. Still manageable for full-batch training on A100.

```python
import torch
from torch_geometric.data import HeteroData

def build_heterogeneous_graph(gene_list, cpg_list, expr_df, meth_df):
    """
    Build a heterogeneous graph with CpG and Gene nodes.

    Parameters
    ----------
    gene_list : list of str
        Gene symbols (5K).
    cpg_list : list of str
        CpG probe IDs (10K).
    expr_df : pd.DataFrame
        Gene expression matrix [987 x 5K].
    meth_df : pd.DataFrame
        CpG methylation matrix [987 x 10K].

    Returns
    -------
    graph_template : dict with edge_index tensors per relation
    gene_to_idx : dict
    cpg_to_idx : dict
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    cpg_to_idx = {c: i for i, c in enumerate(cpg_list)}

    # --- Edge type 1: CpG -> Gene (regulatory) ---
    manifest = pd.read_csv(
        'data/external/HumanMethylation450_15017482_v1-2.csv',
        skiprows=7, low_memory=False,
        usecols=['IlmnID', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group']
    )
    promoter = manifest[manifest['UCSC_RefGene_Group'].str.contains(
        'TSS200|TSS1500|5\'UTR', na=False
    )]
    promoter = promoter[promoter['IlmnID'].isin(cpg_list)]
    promoter = promoter.assign(
        gene=promoter['UCSC_RefGene_Name'].str.split(';')
    ).explode('gene')
    promoter = promoter[promoter['gene'].isin(gene_list)]

    cpg_gene_src = promoter['IlmnID'].map(cpg_to_idx).values
    cpg_gene_dst = promoter['gene'].map(gene_to_idx).values
    edge_index_cpg_gene = torch.tensor(
        np.stack([cpg_gene_src, cpg_gene_dst]), dtype=torch.long
    )

    # --- Edge type 2: Gene <-> Gene (PPI) ---
    edge_index_gene_gene, _ = build_string_ppi_graph(gene_list)

    # --- Edge type 3 (optional): CpG <-> CpG (co-methylation) ---
    # Compute pairwise correlation on a subset for efficiency
    # Use top-k per CpG to keep edges manageable
    corr_matrix = meth_df[cpg_list].corr()
    cpg_src, cpg_dst = [], []
    k = 10  # top-k correlated neighbors per CpG
    for i, cpg in enumerate(cpg_list):
        top_k = corr_matrix[cpg].abs().nlargest(k + 1).index[1:]  # skip self
        for neighbor in top_k:
            if neighbor in cpg_to_idx:
                j = cpg_to_idx[neighbor]
                if corr_matrix.loc[cpg, neighbor] > 0.7:
                    cpg_src.append(i)
                    cpg_dst.append(j)

    edge_index_cpg_cpg = torch.tensor(
        np.stack([cpg_src + cpg_dst, cpg_dst + cpg_src]),
        dtype=torch.long
    ) if cpg_src else torch.zeros(2, 0, dtype=torch.long)

    return {
        ('cpg', 'regulates', 'gene'): edge_index_cpg_gene,
        ('gene', 'regulated_by', 'cpg'): edge_index_cpg_gene.flip(0),
        ('gene', 'interacts', 'gene'): edge_index_gene_gene,
        ('cpg', 'co_methylated', 'cpg'): edge_index_cpg_cpg,
    }, gene_to_idx, cpg_to_idx


def create_hetero_cell_line(cell_row, gene_list, cpg_list,
                            edge_dict, expr_cols, meth_cols):
    """
    Create a HeteroData object for one cell line.
    """
    data = HeteroData()

    # Gene node features: [num_genes, 1]
    gene_feats = torch.tensor(
        [[cell_row[f'expr_{g}']] for g in gene_list],
        dtype=torch.float
    )
    data['gene'].x = gene_feats

    # CpG node features: [num_cpgs, 1]
    cpg_feats = torch.tensor(
        [[cell_row[c]] for c in cpg_list],
        dtype=torch.float
    )
    data['cpg'].x = cpg_feats

    # Edges (shared structure across all cell lines)
    for rel, ei in edge_dict.items():
        data[rel].edge_index = ei

    return data
```

### 5.4 Model Architecture

```python
import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, HeteroConv, GATConv, Linear

class HeterogeneousCellEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer for CpG-Gene graphs.
    Uses HGTConv (Heterogeneous Graph Transformer) which handles
    multiple node types and edge types natively.
    """
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=3,
                 dropout=0.3):
        super().__init__()

        # Node type metadata
        self.node_types = ['gene', 'cpg']
        self.metadata = (
            ['gene', 'cpg'],
            [
                ('cpg', 'regulates', 'gene'),
                ('gene', 'regulated_by', 'cpg'),
                ('gene', 'interacts', 'gene'),
                ('cpg', 'co_methylated', 'cpg'),
            ]
        )

        # Input projections (different feature dims per node type)
        self.gene_proj = nn.Linear(1, hidden_dim)
        self.cpg_proj = nn.Linear(1, hidden_dim)

        # HGT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_dim, hidden_dim, self.metadata,
                        num_heads, group='sum')
            )
            self.norms.append(nn.ModuleDict({
                'gene': nn.LayerNorm(hidden_dim),
                'cpg': nn.LayerNorm(hidden_dim),
            }))

        # Dual-stream attention pooling
        self.gene_pool = AttentionPooling(hidden_dim)
        self.cpg_pool = AttentionPooling(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        """
        Parameters
        ----------
        x_dict : dict of {node_type: [total_nodes, feat_dim]}
        edge_index_dict : dict of {(src, rel, dst): [2, edges]}
        batch_dict : dict of {node_type: [total_nodes]}
        """
        # Project to hidden dim
        h_dict = {
            'gene': self.gene_proj(x_dict['gene']),
            'cpg': self.cpg_proj(x_dict['cpg']),
        }

        # Message passing layers
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h_dict, edge_index_dict)
            h_dict = {
                key: self.dropout(norm[key](h_new[key]).relu()) + h_dict[key]
                for key in h_dict
            }

        # Dual-stream pooling
        h_gene = self.gene_pool(h_dict['gene'], batch_dict['gene'])
        h_cpg = self.cpg_pool(h_dict['cpg'], batch_dict['cpg'])

        # Concatenate streams
        return torch.cat([h_gene, h_cpg], dim=-1)  # [batch, 2*hidden]


class ArchitectureB(nn.Module):
    """
    Architecture B: CpG-Gene Heterogeneous Graph.

    Cell line: HGT on heterogeneous CpG-Gene graph -> dual pooling -> h_cell
    Drug: Pre-trained GIN -> h_drug
    Fusion: Tensor product partial -> MLP -> IC50
    """
    def __init__(self, hidden_dim=128, drug_dim=256, dropout=0.3):
        super().__init__()

        self.cell_encoder = HeterogeneousCellEncoder(
            hidden_dim=hidden_dim, num_heads=4,
            num_layers=3, dropout=dropout
        )

        self.fusion = TensorProductFusion(
            cell_dim=hidden_dim * 2,  # dual stream: gene + cpg
            drug_dim=drug_dim,
            rank=128,
            out_dim=256
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_dict, edge_index_dict, batch_dict, h_drug):
        h_cell = self.cell_encoder(x_dict, edge_index_dict, batch_dict)
        h_fused = self.fusion(h_cell, h_drug)
        return self.head(h_fused).squeeze(-1)
```

### 5.5 Alternative: HeteroConv Wrapper

If HGTConv is too heavy, use `HeteroConv` which wraps standard conv layers per edge type:

```python
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv

class SimplerHeteroEncoder(nn.Module):
    """
    Uses HeteroConv with per-relation convolution layers.
    Lighter than HGT, easier to debug.
    """
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()

        self.gene_proj = nn.Linear(1, hidden_dim)
        self.cpg_proj = nn.Linear(1, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('cpg', 'regulates', 'gene'): SAGEConv(
                    hidden_dim, hidden_dim),
                ('gene', 'regulated_by', 'cpg'): SAGEConv(
                    hidden_dim, hidden_dim),
                ('gene', 'interacts', 'gene'): GATConv(
                    hidden_dim, hidden_dim, heads=4, concat=False),
                ('cpg', 'co_methylated', 'cpg'): SAGEConv(
                    hidden_dim, hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        h = {
            'gene': self.gene_proj(x_dict['gene']),
            'cpg': self.cpg_proj(x_dict['cpg']),
        }
        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            h = {k: self.dropout(v.relu()) + h[k] for k, v in h_new.items()}
        return h
```

### 5.6 Training Considerations

**Missing IC50 data**: The 987 x 375 IC50 matrix has significant sparsity (~15-20% missing). Options:

1. **Train on observed entries only** (simplest, recommended first):
   ```python
   # Only create samples where IC50 is not NaN
   mask = ~drug_response_df.isna()
   ```

2. **Pre-impute with RT-DMF** (recommended for final model):
   ```python
   # Run RT-DMF denoising/imputation as preprocessing
   # pip install rtdmf (or clone from github.com/tomwhoooo/rtdmf)
   from rtdmf import RTDMF
   imputed_ic50 = RTDMF().fit_transform(drug_response_matrix)
   ```

3. **Joint imputation + prediction** (advanced, future work):
   Add a matrix factorization loss alongside the supervised IC50 loss.

### 5.7 Implementation Plan

| Step | Task | Deliverable | Est. Time |
|------|------|-------------|-----------|
| B.1 | Complete all Architecture A data prep (A.1-A.3) | External data downloaded | (shared) |
| B.2 | Build heterogeneous graph construction | `src/graphs/hetero_graph.py` | 3 days |
| B.3 | Compute CpG co-methylation edges | `src/graphs/co_methylation.py` | 2 days |
| B.4 | Write HeteroData PyG dataset class | `src/data/hetero_dataset.py` | 2 days |
| B.5 | Implement HGT cell encoder | `src/models/architecture_b.py` | 3 days |
| B.6 | Implement HeteroConv alternative | Same file, alternative class | 1 day |
| B.7 | Adapt training loop for HeteroData | `src/training/train_hetero.py` | 2 days |
| B.8 | Run ablation: with/without co-methylation edges | `results/arch_b/` | 2 days |
| B.9 | Run ablation: HGT vs HeteroConv | Same | 2 days |
| B.10 | Compare vs Architecture A on all splits | Same | 2 days |
| B.11 | Attention analysis: which CpG→Gene edges are most important? | `results/arch_b/interpretability/` | 2 days |
| **Total** | | | **~21 days** |

### 5.8 Expected Outcomes

- **Random split**: R² 0.20-0.30 (improvement from explicit CpG-gene structure)
- **Histology split**: R² 0.05-0.12 (epigenetic regulatory patterns may transfer better than raw features)
- **Site split**: R² 0.03-0.10
- **Interpretability**: Identify which CpG-gene regulatory edges are most predictive per drug. Novel biological findings possible.
- **Publishable novelty**: First CpG-gene heterogeneous graph for drug response.

---

## 6. Architecture C: Pathway-Hierarchical Graph Transformer + Contrastive Pre-training

### 6.1 Overview

**Objective**: Maximum performance on the cross-cancer generalization problem through: (1) pathway-level abstraction that transfers across cancer types, (2) GPS graph transformer for capturing long-range gene dependencies, and (3) contrastive pre-training that learns cancer-type-invariant representations.

**Novelty claims**:
1. Pathway-hierarchical GNN with GPS layers for drug response
2. Contrastive pre-training objective for cross-cancer generalization
3. Multi-head bilinear drug-pathway attention fusion

**Timeline**: 6-8 weeks.

### 6.2 Architecture Diagram

```
DRUG BRANCH                              CELL LINE BRANCH

SMILES → [GIN] → h_drug [256]           Gene Expression (5K genes)
              │                                 │
              │                           ┌─────▼──────────────────────┐
              │                           │  GENE-LEVEL GPS ENCODER    │
              │                           │                            │
              │                           │  STRING PPI Graph (5K)     │
              │                           │  + LapPE + RWSE positional │
              │                           │                            │
              │                           │  GPS Layer 1: GIN + Attn   │
              │                           │  GPS Layer 2: GIN + Attn   │
              │                           │                            │
              │                           │  → h_genes [5K, 128]       │
              │                           └─────┬──────────────────────┘
              │                                 │
              │                           ┌─────▼──────────────────────┐
              │                           │  PATHWAY AGGREGATION       │
              │                           │                            │
              │                           │  For each pathway p:       │
              │                           │  h_p = Attn_pool(h_genes   │
              │                           │        in pathway p)       │
              │                           │                            │
              │                           │  → h_pathways [~300, 128]  │
              │                           └─────┬──────────────────────┘
              │                                 │
              │                           ┌─────▼──────────────────────┐
              │                           │  PATHWAY GRAPH TRANSFORMER │
              │                           │                            │
              │                           │  Pathway crosstalk graph   │
              │                           │  (shared genes between     │
              │                           │   pathways = edges)        │
              │                           │                            │
              │                           │  GPS Layer: GIN + Attn     │
              │                           │                            │
              │                           │  → h_pathways' [~300, 128] │
              │                           └─────┬──────────────────────┘
              │                                 │
              │                           Mean Pool → h_cell [128]
              │                                 │
              ▼                                 ▼
        ┌──────────────────────────────────────────────┐
        │  MULTI-HEAD BILINEAR DRUG-PATHWAY ATTENTION  │
        │                                              │
        │  For each head k:                            │
        │    attn_k(drug, pathway_j) = h_drug^T W_k    │
        │                              h_pathway_j     │
        │    h_fused_k = Σ_j softmax(attn_k) *         │
        │               h_pathway_j                    │
        │                                              │
        │  h_fused = concat([h_fused_1, ..., h_fused_K])│
        └───────────────────┬──────────────────────────┘
                            │
                      MLP Head → IC50

═══════════════════════════════════════════════════════
   CONTRASTIVE PRE-TRAINING (Phase 1, before IC50):

   Same cell line, different augmentation → positive pair
   Different cell line, same cancer type → soft positive
   Different cancer type → negative

   Loss: NT-Xent (normalized temperature cross-entropy)
═══════════════════════════════════════════════════════
```

### 6.3 Gene-Level GPS Encoder

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

class GeneGPSEncoder(nn.Module):
    """
    Gene-level encoder using GPS (General Powerful Scalable) layers.

    GPS combines:
    1. Local message passing (GIN) for neighborhood structure
    2. Global attention (Transformer) for long-range dependencies
    3. Positional encodings (LapPE + RWSE) for structural awareness
    """
    def __init__(self, in_dim, hidden_dim=128, num_layers=2,
                 num_heads=4, dropout=0.3, pe_dim=16):
        super().__init__()

        self.pe_dim = pe_dim  # Positional encoding dimension

        # Input projection (gene features + positional encodings)
        self.input_proj = nn.Linear(in_dim + pe_dim * 2, hidden_dim)

        # GPS layers
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Local MPNN: GIN
            gin_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            local_mpnn = GINConv(gin_mlp)

            # GPS wraps local MPNN + global attention
            gps = GPSConv(
                channels=hidden_dim,
                conv=local_mpnn,
                heads=num_heads,
                dropout=dropout,
                attn_dropout=dropout,
            )
            self.gps_layers.append(gps)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, pe_lap, pe_rw, batch=None):
        """
        Parameters
        ----------
        x : [num_nodes, in_dim] gene features
        edge_index : [2, E] PPI edges
        pe_lap : [num_nodes, pe_dim] Laplacian PE
        pe_rw : [num_nodes, pe_dim] Random Walk PE
        batch : [num_nodes] graph membership

        Returns
        -------
        h : [num_nodes, hidden_dim] per-gene embeddings
        """
        # Concatenate features + positional encodings
        x = torch.cat([x, pe_lap, pe_rw], dim=-1)
        h = self.input_proj(x)

        # GPS layers
        for gps in self.gps_layers:
            h = gps(h, edge_index, batch=batch)

        return self.norm(h)
```

### 6.4 Pathway Aggregation

```python
class PathwayAggregator(nn.Module):
    """
    Aggregate gene-level embeddings into pathway-level embeddings.

    Uses attention pooling per pathway so the model learns which genes
    within each pathway are most relevant for drug response.
    """
    def __init__(self, hidden_dim=128, num_pathways=300):
        super().__init__()

        # Per-pathway attention (shared weights, pathway-specific query)
        self.pathway_queries = nn.Parameter(
            torch.randn(num_pathways, hidden_dim)
        )
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, h_genes, pathway_masks):
        """
        Parameters
        ----------
        h_genes : [num_genes, hidden_dim]
            Per-gene embeddings from GPS encoder.
        pathway_masks : [num_pathways, num_genes]
            Binary masks: pathway_masks[p, g] = 1 if gene g in pathway p.

        Returns
        -------
        h_pathways : [num_pathways, hidden_dim]
        """
        # Project genes for attention
        h_proj = self.attn_proj(h_genes)  # [G, D]

        pathway_embeddings = []
        for p in range(pathway_masks.size(0)):
            mask = pathway_masks[p].bool()  # [G]
            if mask.sum() == 0:
                pathway_embeddings.append(
                    torch.zeros(h_genes.size(1), device=h_genes.device)
                )
                continue

            # Genes in this pathway
            h_p = h_proj[mask]  # [|pathway|, D]
            query = self.pathway_queries[p]  # [D]

            # Attention: query dot product with gene embeddings
            attn = (h_p @ query) / self.scale  # [|pathway|]
            attn = torch.softmax(attn, dim=0)

            # Weighted sum
            h_pathway = (attn.unsqueeze(-1) * h_genes[mask]).sum(dim=0)
            pathway_embeddings.append(h_pathway)

        return torch.stack(pathway_embeddings)  # [P, D]


def build_pathway_masks(gene_list, pathway_db='kegg'):
    """
    Build binary pathway membership masks.

    Parameters
    ----------
    gene_list : list of str
        Ordered gene symbols.
    pathway_db : str
        'kegg', 'reactome', or 'msigdb_hallmark'.

    Returns
    -------
    pathway_masks : torch.FloatTensor [num_pathways, num_genes]
    pathway_names : list of str
    """
    # Load pathway gene sets (from MSigDB GMT file)
    # Download: https://www.gsea-msigdb.org/gsea/msigdb/
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    pathway_masks = []
    pathway_names = []

    with open(f'data/external/{pathway_db}_pathways.gmt') as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0]
            genes = parts[2:]

            mask = torch.zeros(len(gene_list))
            for gene in genes:
                if gene in gene_to_idx:
                    mask[gene_to_idx[gene]] = 1.0

            if mask.sum() >= 5:  # Skip pathways with <5 genes in our set
                pathway_masks.append(mask)
                pathway_names.append(name)

    return torch.stack(pathway_masks), pathway_names


def build_pathway_crosstalk_graph(pathway_masks, min_shared=3):
    """
    Build pathway-pathway graph based on shared genes.
    Two pathways are connected if they share >= min_shared genes.

    Returns
    -------
    edge_index : [2, E] pathway-level edges
    """
    num_pathways = pathway_masks.size(0)
    src, dst = [], []

    # Compute pairwise overlap
    overlap = pathway_masks @ pathway_masks.T  # [P, P]

    for i in range(num_pathways):
        for j in range(i + 1, num_pathways):
            if overlap[i, j] >= min_shared:
                src.extend([i, j])
                dst.extend([j, i])

    return torch.tensor([src, dst], dtype=torch.long)
```

### 6.5 Multi-Head Bilinear Drug-Pathway Attention

```python
class DrugPathwayAttention(nn.Module):
    """
    Multi-head bilinear attention between drug and pathway embeddings.

    For each attention head, computes:
      attn(drug, pathway_j) = h_drug^T W_k h_pathway_j

    Then aggregates pathway embeddings weighted by drug-specific attention.
    This lets the model learn which pathways matter for each drug.

    Inspired by BANDRP (BIB 2024).
    """
    def __init__(self, drug_dim=256, pathway_dim=128,
                 num_heads=4, out_dim=256):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = pathway_dim // num_heads

        # Per-head bilinear weights
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(drug_dim, self.head_dim) * 0.01)
            for _ in range(num_heads)
        ])

        # Drug projection per head
        self.drug_projs = nn.ModuleList([
            nn.Linear(drug_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # Pathway projection per head
        self.pathway_projs = nn.ModuleList([
            nn.Linear(pathway_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(self.head_dim * num_heads, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU()
        )

    def forward(self, h_drug, h_pathways):
        """
        Parameters
        ----------
        h_drug : [batch, drug_dim]
        h_pathways : [batch, num_pathways, pathway_dim]

        Returns
        -------
        h_fused : [batch, out_dim]
        """
        head_outputs = []

        for k in range(self.num_heads):
            # Project
            d_k = self.drug_projs[k](h_drug)        # [B, head_dim]
            p_k = self.pathway_projs[k](h_pathways)  # [B, P, head_dim]

            # Bilinear attention: drug^T @ W_k @ pathway
            # Simplification: (d_k) dot (p_k) per pathway
            attn = torch.einsum('bd,bpd->bp', d_k, p_k)  # [B, P]
            attn = torch.softmax(attn / (self.head_dim ** 0.5), dim=-1)

            # Weighted sum of pathway embeddings
            h_k = torch.einsum('bp,bpd->bd', attn, p_k)  # [B, head_dim]
            head_outputs.append(h_k)

        # Concatenate heads
        h_multi = torch.cat(head_outputs, dim=-1)  # [B, heads*head_dim]
        return self.output_proj(h_multi)
```

### 6.6 Contrastive Pre-training

```python
class ContrastivePretrainer(nn.Module):
    """
    Contrastive pre-training for cross-cancer generalization.

    Goal: Learn cell line representations where biological similarity
    (pathway activation patterns) matters more than cancer-type identity.

    Strategy:
    - Positive pairs: Same cell line with different feature augmentations
    - Soft positives: Different cell lines with similar pathway profiles
    - Negatives: Random cell lines

    Loss: NT-Xent (InfoNCE variant)
    """
    def __init__(self, encoder, proj_dim=128, temperature=0.1):
        super().__init__()
        self.encoder = encoder  # Gene GPS encoder
        self.projector = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.temperature = temperature

    def augment(self, x, edge_index, drop_rate=0.2):
        """
        Graph augmentation for contrastive learning.

        Augmentation strategies:
        1. Feature masking: Randomly zero out gene features
        2. Edge dropping: Randomly remove PPI edges
        3. Gaussian noise: Add noise to expression values
        """
        # Feature masking
        mask = torch.bernoulli(
            torch.ones_like(x) * (1 - drop_rate)
        )
        x_aug = x * mask

        # Edge dropping
        edge_mask = torch.rand(edge_index.size(1)) > drop_rate
        edge_index_aug = edge_index[:, edge_mask]

        return x_aug, edge_index_aug

    def nt_xent_loss(self, z1, z2):
        """Normalized temperature-scaled cross-entropy loss."""
        batch_size = z1.size(0)

        # Normalize
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Similarity matrix
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = z @ z.T / self.temperature  # [2B, 2B]

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -float('inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)

        return nn.functional.cross_entropy(sim, labels)

    def forward(self, x, edge_index, pe_lap, pe_rw, batch):
        # Two augmented views
        x1, ei1 = self.augment(x, edge_index)
        x2, ei2 = self.augment(x, edge_index)

        # Encode both views
        h1 = self.encoder(x1, ei1, pe_lap, pe_rw, batch)
        h2 = self.encoder(x2, ei2, pe_lap, pe_rw, batch)

        # Pool to graph-level
        from torch_geometric.nn import global_mean_pool
        g1 = global_mean_pool(h1, batch)
        g2 = global_mean_pool(h2, batch)

        # Project
        z1 = self.projector(g1)
        z2 = self.projector(g2)

        return self.nt_xent_loss(z1, z2)
```

### 6.7 Full Architecture C Model

```python
class ArchitectureC(nn.Module):
    """
    Architecture C: Pathway-Hierarchical GPS + Contrastive Pre-training.

    Pipeline:
    1. Gene GPS encoder (local GIN + global attention on PPI graph)
    2. Pathway aggregation (attention pool genes per pathway)
    3. Pathway GPS encoder (capture inter-pathway dependencies)
    4. Drug-pathway bilinear attention fusion
    5. MLP prediction head
    """
    def __init__(self, gene_feat_dim=1, hidden_dim=128, drug_dim=256,
                 num_pathways=300, num_heads=4, dropout=0.3, pe_dim=16):
        super().__init__()

        # Level 1: Gene encoder
        self.gene_encoder = GeneGPSEncoder(
            in_dim=gene_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout,
            pe_dim=pe_dim
        )

        # Level 2: Pathway aggregation
        self.pathway_agg = PathwayAggregator(
            hidden_dim=hidden_dim,
            num_pathways=num_pathways
        )

        # Level 3: Pathway-level GPS
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pathway_gps = GPSConv(
            channels=hidden_dim,
            conv=GINConv(gin_mlp),
            heads=num_heads,
            dropout=dropout
        )
        self.pathway_norm = nn.LayerNorm(hidden_dim)

        # Level 4: Drug-pathway fusion
        self.fusion = DrugPathwayAttention(
            drug_dim=drug_dim,
            pathway_dim=hidden_dim,
            num_heads=num_heads,
            out_dim=256
        )

        # Level 5: Prediction
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_genes, edge_index_ppi, pe_lap, pe_rw,
                pathway_masks, pathway_edge_index,
                h_drug, batch_genes=None, batch_pathways=None):
        """
        Parameters
        ----------
        x_genes : [B*G, feat_dim] gene features (batched)
        edge_index_ppi : [2, E_ppi] PPI edges (batched)
        pe_lap : [B*G, pe_dim] Laplacian PE
        pe_rw : [B*G, pe_dim] Random Walk PE
        pathway_masks : [P, G] pathway membership (shared)
        pathway_edge_index : [2, E_pathway] pathway crosstalk (shared)
        h_drug : [B, drug_dim] drug embeddings
        batch_genes : [B*G] gene-to-graph mapping
        batch_pathways : [B*P] pathway-to-graph mapping
        """
        batch_size = h_drug.size(0)
        num_genes = pathway_masks.size(1)
        num_pathways = pathway_masks.size(0)

        # 1. Gene-level encoding
        h_genes = self.gene_encoder(
            x_genes, edge_index_ppi, pe_lap, pe_rw, batch_genes
        )  # [B*G, hidden]

        # 2. Pathway aggregation (per graph in batch)
        # Reshape to per-graph: [B, G, hidden]
        h_genes_per_graph = h_genes.view(batch_size, num_genes, -1)

        h_pathways_list = []
        for b in range(batch_size):
            h_p = self.pathway_agg(h_genes_per_graph[b], pathway_masks)
            h_pathways_list.append(h_p)
        h_pathways = torch.stack(h_pathways_list)  # [B, P, hidden]

        # 3. Pathway-level GPS (process as batch of pathway graphs)
        h_pw_flat = h_pathways.view(-1, h_pathways.size(-1))  # [B*P, hidden]

        # Expand pathway edges for batch
        pw_edge_list = []
        for b in range(batch_size):
            pw_edge_list.append(pathway_edge_index + b * num_pathways)
        pw_edges_batched = torch.cat(pw_edge_list, dim=1)

        if batch_pathways is None:
            batch_pathways = torch.arange(batch_size).repeat_interleave(
                num_pathways
            ).to(h_drug.device)

        h_pw_flat = self.pathway_gps(
            h_pw_flat, pw_edges_batched, batch=batch_pathways
        )
        h_pw_flat = self.pathway_norm(h_pw_flat)
        h_pathways = h_pw_flat.view(batch_size, num_pathways, -1)

        # 4. Drug-pathway bilinear attention
        h_fused = self.fusion(h_drug, h_pathways)  # [B, 256]

        # 5. Predict
        return self.head(h_fused).squeeze(-1)
```

### 6.8 Two-Phase Training

```python
def train_architecture_c():
    """
    Two-phase training:
    Phase 1: Contrastive pre-training (no drug labels needed)
    Phase 2: Supervised fine-tuning on IC50 prediction
    """

    # ========== PHASE 1: Contrastive Pre-training ==========
    print("Phase 1: Contrastive pre-training...")

    gene_encoder = GeneGPSEncoder(in_dim=1, hidden_dim=128)
    pretrainer = ContrastivePretrainer(gene_encoder, proj_dim=128)

    optimizer_pt = torch.optim.Adam(pretrainer.parameters(), lr=3e-4)

    for epoch in range(100):
        pretrainer.train()
        for batch in cell_line_loader:  # No drug labels needed
            batch = batch.to(device)
            loss = pretrainer(
                batch.x, batch.edge_index,
                batch.pe_lap, batch.pe_rw, batch.batch
            )
            optimizer_pt.zero_grad()
            loss.backward()
            optimizer_pt.step()

    # ========== PHASE 2: Supervised Fine-tuning ==========
    print("Phase 2: Supervised fine-tuning on IC50...")

    model = ArchitectureC(
        gene_feat_dim=1,
        hidden_dim=128,
        drug_dim=256,
        num_pathways=len(pathway_names)
    ).to(device)

    # Load pre-trained gene encoder weights
    model.gene_encoder.load_state_dict(gene_encoder.state_dict())

    # Lower learning rate for pre-trained encoder, higher for new layers
    optimizer = torch.optim.Adam([
        {'params': model.gene_encoder.parameters(), 'lr': 1e-5},
        {'params': model.pathway_agg.parameters(), 'lr': 1e-4},
        {'params': model.pathway_gps.parameters(), 'lr': 1e-4},
        {'params': model.fusion.parameters(), 'lr': 1e-4},
        {'params': model.head.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-5)

    # Standard training with early stopping
    for epoch in range(300):
        model.train()
        for batch in drug_response_loader:
            pred = model(
                batch.x_genes, batch.edge_index_ppi,
                batch.pe_lap, batch.pe_rw,
                pathway_masks, pathway_edge_index,
                drug_embeddings[batch.drug_idx],
                batch.batch_genes, batch.batch_pathways
            )
            loss = nn.functional.mse_loss(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

### 6.9 Implementation Plan

| Step | Task | Deliverable | Est. Time |
|------|------|-------------|-----------|
| C.1 | Complete Architecture A infrastructure (data, splits, eval) | (shared) | (shared) |
| C.2 | Download pathway gene sets (MSigDB KEGG + Hallmark) | `data/external/kegg_pathways.gmt`, `hallmark.gmt` | 1 day |
| C.3 | Build pathway masks and crosstalk graph | `src/graphs/pathways.py` | 2 days |
| C.4 | Implement GPS gene encoder with LapPE + RWSE | `src/models/gps_encoder.py` | 3 days |
| C.5 | Implement pathway aggregation module | `src/models/pathway_agg.py` | 2 days |
| C.6 | Implement pathway-level GPS layer | Same as C.4 | 1 day |
| C.7 | Implement drug-pathway bilinear attention | `src/models/drug_pathway_attn.py` | 2 days |
| C.8 | Implement contrastive pre-training pipeline | `src/training/pretrain_contrastive.py` | 3 days |
| C.9 | Implement two-phase training with LR scheduling | `src/training/train_arch_c.py` | 2 days |
| C.10 | Run contrastive pre-training (Phase 1) | `models/pretrained/` | 2 days |
| C.11 | Run supervised fine-tuning (Phase 2) | `results/arch_c/` | 3 days |
| C.12 | Ablation: with/without contrastive pre-training | Same | 2 days |
| C.13 | Ablation: GPS vs GAT at gene level | Same | 2 days |
| C.14 | Ablation: pathway hierarchy vs flat gene pool | Same | 2 days |
| C.15 | Drug-pathway attention visualization | `results/arch_c/interpretability/` | 2 days |
| C.16 | Compare vs Architectures A and B on all splits | `results/comparison/` | 2 days |
| **Total** | | | **~31 days** |

### 6.10 Expected Outcomes

- **Random split**: R² 0.25-0.35 (strong performance from pathway structure + pre-training)
- **Histology split**: R² 0.10-0.18 (contrastive pre-training specifically targets this)
- **Site split**: R² 0.08-0.15 (same reasoning)
- **Drug interpretation**: Drug-pathway attention maps reveal which biological pathways each drug targets, validatable against known mechanism of action
- **Pathway importance**: Which pathways are most predictive across drugs (potential biological discovery)

---

## 7. Shared Components

### 7.1 Evaluation Framework

All architectures share the same evaluation pipeline for fair comparison.

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, loader, drug_embeddings, device='cuda'):
    """
    Compute comprehensive metrics for drug response prediction.

    Returns
    -------
    metrics : dict with keys:
        'mse', 'rmse', 'r2', 'pearson_r', 'spearman_r',
        'per_drug_r2', 'per_drug_pearson'
    """
    model.eval()
    all_preds, all_labels, all_drugs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(
                batch.x, batch.edge_index, batch.batch,
                drug_embeddings[batch.drug_idx].to(device)
            )
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
            all_drugs.append(batch.drug_idx.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    drugs = np.concatenate(all_drugs)

    # Global metrics
    metrics = {
        'mse': mean_squared_error(labels, preds),
        'rmse': np.sqrt(mean_squared_error(labels, preds)),
        'r2': r2_score(labels, preds),
        'pearson_r': pearsonr(labels, preds)[0],
        'spearman_r': spearmanr(labels, preds)[0],
    }

    # Per-drug metrics
    per_drug_r2 = {}
    per_drug_pearson = {}
    for drug_idx in np.unique(drugs):
        mask = drugs == drug_idx
        if mask.sum() >= 5:
            per_drug_r2[drug_idx] = r2_score(labels[mask], preds[mask])
            per_drug_pearson[drug_idx] = pearsonr(
                labels[mask], preds[mask]
            )[0]

    metrics['per_drug_r2'] = per_drug_r2
    metrics['per_drug_pearson'] = per_drug_pearson
    metrics['mean_per_drug_r2'] = np.mean(list(per_drug_r2.values()))

    return metrics


def compare_architectures(results_dict):
    """
    Print comparison table across architectures and splits.

    Parameters
    ----------
    results_dict : dict of {arch_name: {split_name: metrics}}
    """
    splits = ['random', 'histology', 'site']
    archs = list(results_dict.keys())

    print(f"{'Architecture':<20} {'Random R²':>10} {'Histology R²':>13} "
          f"{'Site R²':>10} {'Random PCC':>11}")
    print("-" * 70)

    for arch in archs:
        row = f"{arch:<20}"
        for split in splits:
            r2 = results_dict[arch][split]['r2']
            row += f" {r2:>10.4f}"
        pcc = results_dict[arch]['random']['pearson_r']
        row += f" {pcc:>11.4f}"
        print(row)
```

### 7.2 Split Strategies

```python
from sklearn.model_selection import GroupKFold

def create_splits(df, split_type='histology', seed=42):
    """
    Create train/val/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-omics DataFrame with metadata columns.
    split_type : str
        'random', 'histology', or 'site'.

    Returns
    -------
    train_idx, val_idx, test_idx : arrays of sample indices
    """
    np.random.seed(seed)

    if split_type == 'random':
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(df))
        train_val, test = train_test_split(idx, test_size=0.2, random_state=seed)
        train, val = train_test_split(train_val, test_size=0.125,
                                       random_state=seed)
        return train, val, test

    elif split_type == 'histology':
        groups = df['primary histology'].values
        unique_groups = np.unique(groups)
        np.random.shuffle(unique_groups)

        # Hold out ~20% of histology groups for test, ~10% for val
        n_test = max(1, int(len(unique_groups) * 0.2))
        n_val = max(1, int(len(unique_groups) * 0.1))

        test_groups = set(unique_groups[:n_test])
        val_groups = set(unique_groups[n_test:n_test + n_val])

        test_idx = np.where(np.isin(groups, list(test_groups)))[0]
        val_idx = np.where(np.isin(groups, list(val_groups)))[0]
        train_idx = np.where(
            ~np.isin(groups, list(test_groups | val_groups))
        )[0]

        return train_idx, val_idx, test_idx

    elif split_type == 'site':
        groups = df['primary site'].values
        unique_groups = np.unique(groups)
        np.random.shuffle(unique_groups)

        n_test = max(1, int(len(unique_groups) * 0.2))
        n_val = max(1, int(len(unique_groups) * 0.1))

        test_groups = set(unique_groups[:n_test])
        val_groups = set(unique_groups[n_test:n_test + n_val])

        test_idx = np.where(np.isin(groups, list(test_groups)))[0]
        val_idx = np.where(np.isin(groups, list(val_groups)))[0]
        train_idx = np.where(
            ~np.isin(groups, list(test_groups | val_groups))
        )[0]

        return train_idx, val_idx, test_idx
```

### 7.3 Data Preprocessing Pipeline

```python
def preprocess_for_gnn(omics_path, drug_response_path):
    """
    End-to-end preprocessing from raw CSVs to GNN-ready data.

    Steps:
    1. Load multi-omics data
    2. Select top 5K variable genes with expression data
    3. Map CpG sites to genes
    4. Standardize features
    5. Build graph structures
    6. Create PyG dataset
    """
    # 1. Load data
    df = pd.read_csv(omics_path, index_col=0)
    drug_df = pd.read_csv(drug_response_path, index_col=0)

    # 2. Get gene and CpG columns
    expr_cols = [c for c in df.columns if c.startswith('expr_')]
    meth_cols = [c for c in df.columns if c.startswith('cg')]
    gene_symbols = [c.replace('expr_', '') for c in expr_cols]

    # Select top 5K most variable genes
    gene_var = df[expr_cols].var().sort_values(ascending=False)
    top_genes = [c.replace('expr_', '') for c in gene_var.head(5000).index]

    # 3. Standardize expression features (per gene across cell lines)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[expr_cols] = scaler.fit_transform(df[expr_cols])
    # Methylation beta values are already in [0,1], no scaling needed

    # 4. Build graph
    edge_index, gene_to_idx = build_string_ppi_graph(top_genes)
    cpg_gene_map = map_cpg_to_genes(meth_cols, top_genes)

    print(f"Graph: {len(top_genes)} genes, "
          f"{edge_index.size(1)} PPI edges")
    print(f"CpG-Gene mappings: {len(cpg_gene_map)} genes have promoter CpGs")

    return df, drug_df, top_genes, edge_index, cpg_gene_map
```

### 7.4 Required Dependencies

```
# requirements_gnn.txt
# Core
torch>=2.1.0
torch-geometric>=2.5.0
torch-scatter
torch-sparse

# Drug encoding
rdkit-pypi>=2023.9.1

# Biological data
mygene>=3.2.0

# Standard ML
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1

# Optional
rtdmf  # for IC50 imputation
```

---

## 8. Architecture Comparison

### Feature Comparison

| Feature | Architecture A | Architecture B | Architecture C |
|---------|---------------|----------------|----------------|
| **Cell encoder** | 3-layer GAT | 3-layer HGT | 2-layer GPS + pathway GPS |
| **Graph nodes** | 5K genes | 15K (10K CpG + 5K gene) | 5K genes + ~300 pathways |
| **Graph edges** | STRING PPI only | PPI + CpG→Gene + co-meth | PPI + pathway crosstalk |
| **Drug encoder** | Pre-trained GIN | Pre-trained GIN | Pre-trained GIN |
| **Fusion** | Tensor product partial | Tensor product partial | Multi-head bilinear attention |
| **Pre-training** | None | None | Contrastive (NT-Xent) |
| **CpG handling** | Averaged into gene features | Explicit CpG nodes | Averaged into gene features |
| **Novelty** | Engineering baseline | CpG-gene graph (novel) | Hierarchy + contrastive |
| **Interpretability** | GAT attention weights | CpG→Gene edge importance | Drug-pathway attention maps |
| **Parameters (est.)** | ~500K | ~1.5M | ~2M |
| **Memory (A100)** | ~2GB | ~6GB | ~4GB |
| **Training time (est.)** | ~1 hr/epoch | ~3 hr/epoch | ~2 hr/epoch (+ pre-train) |

### Expected Performance

| Split | Baselines | Arch A | Arch B | Arch C |
|-------|-----------|--------|--------|--------|
| **Random** | 0.05-0.10 | 0.15-0.25 | 0.20-0.30 | 0.25-0.35 |
| **Histology** | -0.06 | 0.02-0.08 | 0.05-0.12 | 0.10-0.18 |
| **Site** | -0.02 | 0.01-0.06 | 0.03-0.10 | 0.08-0.15 |

### Risk Assessment

| Risk | Arch A | Arch B | Arch C |
|------|--------|--------|--------|
| **Implementation complexity** | Low | Medium | High |
| **Debugging difficulty** | Low | Medium (hetero batching) | High (multi-stage) |
| **Risk of no improvement** | Medium | Low-Medium | Low |
| **Data requirements** | Low (STRING only) | Medium (manifest + corr) | High (pathways + pre-train) |
| **Publication potential** | Low (incremental) | High (novel gap) | High (if results strong) |

---

## 9. Recommended Progression

### Phase 1: Foundation (Weeks 1-3)

Build Architecture A as the working baseline. This establishes:
- Data pipeline (graph construction, PyG dataset, splits)
- Training infrastructure (training loop, evaluation, logging)
- Drug encoder
- Whether graph structure helps at all

**Key decision point**: If Architecture A shows R² > 0 on histology split, proceed. If not, investigate data quality and graph construction before adding complexity.

### Phase 2: Novel Architecture (Weeks 3-7)

Build Architecture B (CpG-Gene Heterogeneous Graph). This is the primary paper contribution:
- Novel graph structure (CpG-gene regulatory edges)
- Ablation studies proving each edge type's contribution
- Interpretability analysis of CpG-gene regulatory importance

**Key decision point**: Compare Arch B vs Arch A. If heterogeneous graph improves histology/site splits, this confirms the novelty claim.

### Phase 3: Maximum Performance (Weeks 6-10)

Add Architecture C elements incrementally on top of what works:
- Replace GAT with GPS (if Arch A showed attention was useful)
- Add pathway hierarchy (if Arch B showed gene-level patterns)
- Add contrastive pre-training (if generalization is still weak)

**Do NOT build Architecture C from scratch.** Build it by upgrading Architecture A/B with the components that add the most value.

### Ablation Study Matrix

Run these ablations to understand what drives performance:

| Ablation | What It Tests |
|----------|--------------|
| GAT vs GCN vs GIN (gene encoder) | Does attention matter? |
| With vs without CpG-gene edges | Does methylation graph structure help? |
| With vs without co-methylation edges | Do CpG-CpG relationships matter? |
| Tensor product vs concatenation fusion | Does multiplicative fusion help? |
| GIN drug encoder vs Morgan fingerprint MLP | Does drug graph structure matter? |
| With vs without contrastive pre-training | Does pre-training improve generalization? |
| Pathway hierarchy vs flat gene pool | Does biological hierarchy help? |
| GPS vs GAT at gene level | Does global attention help? |

---

## 10. References

### Drug Response GNNs

1. **DRPreter** - Shin et al. (2022). "Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided GNNs and Transformer." IJMS. [PMC9699175](https://pmc.ncbi.nlm.nih.gov/articles/PMC9699175/)
2. **TransCDR** - Xia et al. (2024). "Deep Learning Model for Drug Activity Prediction." BMC Biology. [DOI](https://link.springer.com/article/10.1186/s12915-024-02023-8)
3. **GraphTCDR** - (2025). "Heterogeneous Graph Neural Networks for Drug Response." Neural Networks. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025008822)
4. **PASO** - (2025). "Anticancer Drug Response Prediction." PLOS Computational Biology. [DOI](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012905)
5. **BANDRP** - (2024). "Bilinear Attention Network for Drug Response Prediction." Briefings in Bioinformatics. [DOI](https://academic.oup.com/bib/article/25/6/bbae493/7823034)
6. **XGDP** - (2025). "Explainable Graph Drug Prediction." Nature Scientific Reports. [DOI](https://www.nature.com/articles/s41598-024-83090-3)
7. **GPDRP** - (2023). "Multimodal Framework for Drug Response Prediction with Graph Transformer." BMC Bioinformatics. [DOI](https://link.springer.com/article/10.1186/s12859-023-05618-0)

### Graph Transformer Architectures

8. **GPS** - Rampasek et al. (2022). "Recipe for a General, Powerful, Scalable Graph Transformer." NeurIPS. [arXiv:2205.12454](https://arxiv.org/abs/2205.12454)
9. **Graphormer** - Ying et al. (2021). "Do Transformers Really Perform Bad for Graph Representation?" NeurIPS. [Microsoft](https://www.microsoft.com/en-us/research/project/graphormer/)
10. **Graphormer Boosted** - (2025). "Enhanced Graph Spatial and Edge Encodings." IEEE Access. [DOI](https://ui.adsabs.harvard.edu/abs/2025IEEEA..13m0492F/abstract)

### Fusion Methods

11. **Tensor Product Fusion** - (2024). "Optimal Fusion of Genotype and Drug Embeddings." Briefings in Bioinformatics. [DOI](https://academic.oup.com/bib/article/25/3/bbae227/7675149)

### Pre-trained Molecular Encoders

12. **Uni-Mol2** - (2024). "Exploring Molecular Pretraining Model at Scale." [arXiv](https://arxiv.org/html/2406.14969v1)

### Missing Data / Imputation

13. **RT-DMF** - (2025). "Large-Scale Information Retrieval and Correction." Briefings in Bioinformatics. [DOI](https://academic.oup.com/bib/article/26/3/bbaf226/8150964)
14. **DBDNMF** - (2024). "Dual Branch Deep Neural Matrix Factorization." PLOS Computational Biology. [DOI](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012012)

### Methylation GNNs (Not Drug Response - Gap)

15. **GraphMeXplain** - (2026). Methylation-based graph for explainable prediction. bioRxiv.
16. **GraphAge** - (2025). "Graph-based methylation age prediction." PNAS Nexus.

### Other

17. **metaDRP** - (2025). "Fast-Adapting GNN with Prior Knowledge." CSBT. [DOI](https://www.sciencedirect.com/science/article/pii/S2095177925002035)
18. **CRISP** - (2025). "Predicting Drug Responses of Unseen Cell Types." Nature Computational Science. [DOI](https://www.nature.com/articles/s43588-025-00887-6)
19. **EGNF** - (2025). "Expression Graph Network Framework." Briefings in Bioinformatics. [DOI](https://academic.oup.com/bib/article/26/5/bbaf559/8303424)

---

## Document History

| Date | Changes |
|------|---------|
| 2026-02-07 | Initial compilation from research session. Three architectures with full implementation plans. |

---

*This document serves as the technical reference for GNN architecture design and implementation. Start with Architecture A, validate with Architecture B for the novelty paper, and add Architecture C elements incrementally based on ablation results.*
