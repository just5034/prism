# Graph Neural Networks: An Intuitive Guide for Architecture Design

**Prepared for**: Pharmacoepigenomics Biomarker Discovery Team  
**Context**: Multi-omics drug response prediction (DNA methylation + gene expression → IC50)  
**Date**: December 2025

---

## Table of Contents

1. [Why Graphs? The Motivation](#1-why-graphs-the-motivation)
2. [The Core Intuition: Message Passing](#2-the-core-intuition-message-passing)
3. [Anatomy of a GNN Layer](#3-anatomy-of-a-gnn-layer)
4. [Key GNN Variants](#4-key-gnn-variants)
5. [Graph-Level vs Node-Level Tasks](#5-graph-level-vs-node-level-tasks)
6. [Considerations for Your Problem](#6-considerations-for-your-problem)
7. [Quick Reference: Choosing Your Architecture](#7-quick-reference-choosing-your-architecture)

---

## 1. Why Graphs? The Motivation

### The Limitation of Traditional ML

Your current baseline models (XGBoost, Random Forest) treat each feature independently. When you feed in 10,000 CpG sites + 5,000 genes, the model sees a flat vector:

```
[cg00944421, cg14557185, ..., expr_TP53, expr_EGFR, ...]
     ↓           ↓              ↓           ↓
   0.995       0.999          7.59        8.47
```

**What's lost?** The biological relationships:
- TP53 regulates hundreds of downstream genes
- CpG sites in promoter regions directly affect nearby gene expression
- Genes in the same pathway respond to drugs together

### What Graphs Capture

A graph encodes **entities** (nodes) and **relationships** (edges):

```
        [Gene A]
           │
    regulates
           │
           ▼
        [Gene B] ←── same_pathway ──→ [Gene C]
           │
    promoter_of
           │
           ▼
       [CpG Site]
```

**Key insight**: In a GNN, information flows along edges. A gene's representation is informed not just by its own expression, but by its biological context—what regulates it, what it regulates, what pathway it belongs to.

### Why This Matters for Drug Response

Drugs don't target isolated molecules—they perturb **networks**:
- A kinase inhibitor affects all downstream signaling
- Methylation changes in a promoter affect that gene AND its targets
- Cancer subtypes have distinct pathway dysregulation patterns

**Your "harder" generalization problem** (predicting on unseen histologies/tissues) becomes more tractable if the model learns pathway-level patterns rather than memorizing individual feature correlations.

---

## 2. The Core Intuition: Message Passing

### The Fundamental Idea

Every GNN operates on the same principle: **nodes learn by aggregating information from their neighbors**.

Think of it like a game of telephone, but structured:

```
Round 1: Every node collects messages from immediate neighbors
Round 2: Every node collects messages from neighbors (who now contain info from THEIR neighbors)
Round 3: ...and so on
```

After K rounds, each node's representation contains information from nodes up to K hops away.

### A Concrete Example

Imagine a small gene regulatory network:

```
      [EGFR]           Initial features: expression levels
        │              EGFR: 8.5, MYC: 9.2, TP53: 6.1, CDKN1A: 4.3
        ▼
      [MYC] ──────→ [CDKN1A]
        │              ▲
        ▼              │
      [TP53] ──────────┘
```

**Before GNN**: Each gene is just its expression value.

**After 1 GNN layer**:
- EGFR's representation now includes info about MYC
- MYC's representation includes EGFR + TP53 + CDKN1A
- TP53's representation includes MYC + CDKN1A

**After 2 GNN layers**:
- EGFR now "knows about" TP53 and CDKN1A (through MYC)
- The representation captures 2-hop pathway structure

### The Mathematical Skeleton

At its core, one GNN layer does three things:

```
1. MESSAGE:   Each neighbor sends a message to the central node
2. AGGREGATE: Combine all incoming messages (sum, mean, max, attention-weighted)
3. UPDATE:    Transform the aggregated message into a new node representation
```

Symbolically:

```
h_v^(k+1) = UPDATE( h_v^(k), AGGREGATE({ MESSAGE(h_u^(k)) : u ∈ neighbors(v) }) )
```

Where:
- `h_v^(k)` = representation of node v at layer k
- `neighbors(v)` = nodes connected to v
- Everything is parameterized by learnable weights

---

## 3. Anatomy of a GNN Layer

### Breaking Down the Components

```
┌─────────────────────────────────────────────────────────────┐
│                      GNN LAYER                               │
│                                                              │
│   ┌──────────┐    ┌─────────────┐    ┌──────────┐          │
│   │  MESSAGE │ →  │  AGGREGATE  │ →  │  UPDATE  │          │
│   │ (per edge)│    │ (per node)  │    │(per node)│          │
│   └──────────┘    └─────────────┘    └──────────┘          │
│                                                              │
│   "What info      "How to combine    "How to transform      │
│    to send?"       messages?"         into new features?"   │
└─────────────────────────────────────────────────────────────┘
```

### Message Function Options

| Type | Description | When to Use |
|------|-------------|-------------|
| **Identity** | Send neighbor's features as-is | Simple, when edges are homogeneous |
| **Linear transform** | `W · h_neighbor` | Standard choice, adds capacity |
| **Edge-conditioned** | Different transform per edge type | Multi-relational graphs (e.g., "regulates" vs "inhibits") |
| **Attention-based** | Weight message by learned importance | When some neighbors matter more |

### Aggregation Function Options

| Type | Formula | Properties |
|------|---------|------------|
| **Sum** | `Σ messages` | Preserves neighbor count; most expressive (GIN paper) |
| **Mean** | `(1/N) Σ messages` | Normalized; good when degree varies wildly |
| **Max** | `max(messages)` | Captures "strongest" signal; robust to noise |
| **Attention** | `Σ α_i · messages` | Learned weighting; adaptive |

**Key insight from theory**: Sum aggregation is the most expressive—it can distinguish graph structures that mean/max cannot. But in practice, the choice depends on your data.

### Update Function Options

Typically a neural network applied after aggregation:

```python
# Simple linear + activation
h_new = ReLU(W @ aggregated_message)

# With residual connection (helps deep networks)
h_new = ReLU(W @ aggregated_message) + h_old

# With layer normalization
h_new = LayerNorm(ReLU(W @ aggregated_message) + h_old)
```

---

## 4. Key GNN Variants

### 4.1 Graph Convolutional Network (GCN)

**Paper**: Kipf & Welling, 2016

**Core idea**: Simplify spectral graph convolutions into a single matrix multiplication.

**The formula**:
```
H^(k+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(k) W^(k) )
```

**In plain English**:
- Take the adjacency matrix (who connects to whom)
- Normalize it (so high-degree nodes don't dominate)
- Multiply with node features and a weight matrix
- Apply activation (ReLU)

**Intuition**: Each node's new features = weighted average of neighbor features, transformed.

```python
# Pseudocode for one GCN layer
def gcn_layer(A_normalized, H, W):
    # A_normalized: preprocessed adjacency (includes self-loops)
    # H: node features [num_nodes × feature_dim]
    # W: learnable weights [feature_dim × output_dim]
    return relu(A_normalized @ H @ W)
```

**Strengths**:
- Simple, efficient, well-understood
- Good baseline for any graph task

**Limitations**:
- All neighbors weighted equally (after normalization)
- Transductive in original form (needs full graph at train time)
- Over-smoothing with many layers (all nodes converge to similar representations)

---

### 4.2 GraphSAGE

**Paper**: Hamilton et al., 2017

**Core idea**: Learn to *sample and aggregate* from neighbors—enables inductive learning (generalize to unseen nodes).

**Key innovation**: Instead of using the full neighborhood, sample a fixed number of neighbors per layer.

```
Layer 1: Sample 10 neighbors
Layer 2: Sample 5 neighbors of each sampled neighbor
Total: 10 × 5 = 50 nodes involved per target node
```

**Aggregator options**:
```python
# Mean aggregator
h_neighbors = mean([h_u for u in sampled_neighbors])

# LSTM aggregator (treats neighbors as sequence)
h_neighbors = LSTM([h_u for u in sampled_neighbors])

# Pooling aggregator
h_neighbors = max([MLP(h_u) for u in sampled_neighbors])
```

**Update**: Concatenate self-features with aggregated neighbor features:
```python
h_new = relu(W @ concat([h_self, h_neighbors]))
```

**Strengths**:
- **Inductive**: Can embed new nodes without retraining
- **Scalable**: Fixed computation cost via sampling
- **Flexible**: Multiple aggregator choices

**Limitations**:
- Sampling introduces variance
- Loses some neighbor information
- Aggregator choice matters significantly

**Relevance to your project**: If you frame each cell line as a graph (genes as nodes), GraphSAGE lets you naturally handle new cell lines at inference time.

---

### 4.3 Graph Attention Network (GAT)

**Paper**: Veličković et al., 2017

**Core idea**: Not all neighbors are equally important—learn attention weights for each edge.

**The attention mechanism**:
```
1. Compute attention coefficient for edge (i,j):
   e_ij = LeakyReLU( a^T · [W·h_i || W·h_j] )

2. Normalize across neighbors:
   α_ij = softmax_j(e_ij)

3. Aggregate with attention:
   h_i' = σ( Σ_j α_ij · W · h_j )
```

**Visual intuition**:

```
        [Gene A]
       α=0.7│         α=0.1
            ▼          │
         [Gene B] ◄────┘
       α=0.2│
            ▼
        [Gene C]

Gene B's representation is 70% influenced by Gene A,
only 10% by the other neighbor, 20% by Gene C.
```

**Multi-head attention**: Run K independent attention mechanisms, concatenate results:
```python
h_i' = concat([head_1(h_i), head_2(h_i), ..., head_K(h_i)])
```

**Strengths**:
- Adaptive neighbor weighting (learns what's important)
- Handles heterogeneous neighbor importance naturally
- Interpretable (attention weights show what the model focuses on)

**Limitations**:
- More parameters than GCN
- Attention computation is O(E) where E = edges
- Attention weights can be misleading for interpretation

**Relevance to your project**: If some gene-gene relationships are more predictive of drug response than others, GAT can learn this automatically.

---

### 4.4 Graph Isomorphism Network (GIN)

**Paper**: Xu et al., 2018

**Core idea**: Maximize expressive power to match the Weisfeiler-Lehman graph isomorphism test.

**Key insight**: Most GNNs with mean/max aggregation cannot distinguish certain graph structures. Sum aggregation + MLP achieves maximum discriminative power.

**The formula**:
```
h_v^(k+1) = MLP( (1 + ε) · h_v^(k) + Σ_{u ∈ N(v)} h_u^(k) )
```

Where:
- `ε` is a learnable scalar (or fixed small value)
- Sum aggregation preserves multiset information
- MLP provides universal approximation

**Why sum matters**:

```
Consider two nodes with different neighborhoods:

Node A neighbors: [1, 1, 2]     → Mean = 1.33, Sum = 4
Node B neighbors: [1, 2, 2, 2]  → Mean = 1.75, Sum = 7

Mean: Different ✓
Sum: Different ✓

Node C neighbors: [1, 2]        → Mean = 1.5, Sum = 3  
Node D neighbors: [1, 1, 2, 2]  → Mean = 1.5, Sum = 6

Mean: Same ✗ (can't distinguish)
Sum: Different ✓
```

**Strengths**:
- Theoretically most powerful (among 1-WL equivalent models)
- Simple architecture
- Strong empirical performance on graph classification

**Limitations**:
- Sum aggregation can be sensitive to graph size
- May overfit on smaller graphs
- 1-WL is still limited (can't distinguish some non-isomorphic graphs)

**Relevance to your project**: If you're doing graph-level prediction (one graph per cell line → one IC50), GIN is a strong choice for the encoder.

---

## 5. Graph-Level vs Node-Level Tasks

### Your Task: Graph-Level Regression

You want to predict IC50 (a single number) from an entire cell line's molecular profile.

**Two main approaches**:

#### Approach A: One Graph Per Cell Line

```
Cell Line "A549" (lung cancer)

Nodes: genes + CpG sites (15,000 nodes)
Edges: regulatory relationships, promoter associations, pathway membership
Node features: expression values, methylation beta values

        [TP53]──────[MDM2]
          │           │
        [CpG_x]     [CpG_y]
          │
        [CDKN1A]

               │
               ▼
        ┌──────────────┐
        │  GNN Encoder │ (multiple layers)
        └──────────────┘
               │
               ▼
        ┌──────────────┐
        │   Readout    │ (pool node embeddings)
        └──────────────┘
               │
               ▼
           IC50 = 3.54
```

#### Approach B: Single Large Graph with Cell Line Nodes

```
One big biological knowledge graph
+ Cell line nodes connected to their expressed genes

        [Cell: A549]────[EGFR high]────[Gene: EGFR]
              │                              │
        [Cell: HeLa]────[TP53 mut]─────[Gene: TP53]
                                             │
                                      [Pathway: Apoptosis]

Each cell line node gets an embedding from GNN.
Predict IC50 from that embedding.
```

### Readout Functions (Graph → Vector)

After running GNN layers, you need to collapse all node embeddings into one graph embedding:

| Method | Formula | Notes |
|--------|---------|-------|
| **Mean pooling** | `mean(all node embeddings)` | Simple, treats all nodes equally |
| **Sum pooling** | `sum(all node embeddings)` | Preserves graph size information |
| **Max pooling** | `max(all node embeddings)` | Captures strongest signals |
| **Attention pooling** | `sum(α_i · h_i)` where α learned | Learns which nodes matter for prediction |
| **Set2Set** | LSTM over node set | Powerful but complex |
| **Hierarchical (DiffPool)** | Learned soft clustering | Multi-resolution; captures hierarchy |

**For drug response**: Attention pooling is attractive—it can learn which genes/CpGs are most predictive for each drug.

---

## 6. Considerations for Your Problem

### 6.1 What Should Be Your Graph?

**Option 1: Gene Regulatory Network**
- Nodes: Genes
- Edges: Regulatory relationships (from databases like TRRUST, RegNetwork)
- Node features: Expression values
- Separate: Handle methylation as node features on promoter-associated genes

**Option 2: Gene-CpG Bipartite + Gene Network**
- Nodes: Genes AND CpG sites
- Edges: CpG-Gene (promoter association) + Gene-Gene (pathways/PPI)
- Node features: Expression for genes, beta values for CpGs

**Option 3: Pathway-Level Graph**
- Nodes: Pathways (e.g., KEGG, Reactome)
- Edges: Pathway crosstalk
- Node features: Aggregated expression/methylation of member genes
- Benefit: Much smaller graph (~300 nodes vs 15,000)

**Option 4: Multi-Omics Heterogeneous Graph**
- Multiple node types: Genes, CpGs, Drugs, Cell Lines
- Multiple edge types: regulates, methylates, targets, tested_on
- Use Relational GCN (R-GCN) or heterogeneous attention

### 6.2 Edge Construction Strategies

| Source | Edge Type | Pros | Cons |
|--------|-----------|------|------|
| **STRING database** | Protein-protein interaction | High coverage | Noisy, not directional |
| **TRRUST** | TF → target gene | Curated, directional | Limited coverage |
| **KEGG pathways** | Co-membership | Interpretable | Arbitrary grouping |
| **Correlation** | High correlation in your data | Data-driven | May overfit; not causal |
| **Promoter proximity** | CpG within 2kb of TSS | Mechanistic | Many-to-many |

**Recommendation for first iteration**: Start with STRING PPI edges (well-established, reasonable coverage) + KEGG pathway edges. You can add complexity later.

### 6.3 Handling the Multi-Task Problem

You're predicting IC50 for ~265 drugs. Options:

**A. Shared encoder, separate prediction heads**
```
                    ┌→ [Drug1 Head] → IC50_drug1
Graph → [GNN] → h → ├→ [Drug2 Head] → IC50_drug2
                    └→ [Drug265 Head] → IC50_drug265
```
- Many parameters; may overfit on drugs with few samples

**B. Drug as input (conditional prediction)**
```
Graph → [GNN] → h_cell
                    ↘
                      [Concat/Attention] → [MLP] → IC50
                    ↗
Drug → [Drug Encoder] → h_drug
```
- Single output head; drug embedding informs prediction
- Can generalize to unseen drugs (if drug features available)

**C. Drug node in the graph (heterogeneous)**
```
[Gene nodes] ←──targets──→ [Drug node]
       ↓
    [GNN with drug-gene edges]
       ↓
   [Readout from drug node]
       ↓
      IC50
```
- Naturally incorporates drug-target knowledge
- More complex to set up

### 6.4 The Over-Smoothing Problem

With many GNN layers, node representations converge (become indistinguishable).

**Mitigations**:
- Keep layers shallow (2-4 typical)
- Residual connections: `h_new = GNN(h) + h`
- Jumping knowledge: Concatenate representations from all layers
- DropEdge: Randomly drop edges during training
- PairNorm / NodeNorm: Normalization schemes

### 6.5 Your "Harder" Generalization Split

Your histology-based and site-based splits test:
- Can the model generalize to unseen cancer subtypes?
- Can the model generalize to unseen tissue origins?

**Why GNNs might help**: If the model learns pathway-level patterns (e.g., "high EGFR pathway activity + low DNA repair pathway → sensitive to Drug X"), these patterns may transfer across cancer types.

**What to watch for**: 
- If GNN significantly outperforms baselines on these splits, it suggests the graph structure encodes transferable biology
- If GNN only matches baselines, the graph may not be capturing useful relationships

---

## 7. Quick Reference: Choosing Your Architecture

### Decision Tree for First Iteration

```
Q1: Do you want to use known biological networks?
    ├─ YES → Start with GCN or GAT on STRING/KEGG graph
    └─ NO  → Consider learning graph structure (more advanced)

Q2: Is interpretability important now?
    ├─ YES → Use GAT (attention weights) or GNNExplainer post-hoc
    └─ NO  → GIN or GraphSAGE may give better raw performance

Q3: How big is your graph per sample?
    ├─ Small (<1000 nodes) → Full-batch GCN/GAT feasible
    └─ Large (>5000 nodes) → Consider GraphSAGE sampling or pathway-level

Q4: How are you handling multiple drugs?
    ├─ Separate heads → Shared GNN encoder, drug-specific MLPs
    └─ Drug as input → Drug embedding concatenated with graph embedding
```

### Suggested First Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    FIRST ITERATION PROPOSAL                 │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  GRAPH CONSTRUCTION:                                        │
│  - Nodes: 5,000 most variable genes                        │
│  - Edges: STRING PPI (confidence > 700)                    │
│  - Node features: Expression + avg methylation of          │
│                   promoter CpGs per gene                   │
│                                                             │
│  ENCODER:                                                   │
│  - 3-layer GAT (hidden dim 128, 4 attention heads)         │
│  - Residual connections                                     │
│  - LayerNorm after each layer                              │
│                                                             │
│  READOUT:                                                   │
│  - Attention pooling over gene nodes                       │
│                                                             │
│  PREDICTION:                                                │
│  - Drug embedding (learned, dim 64)                        │
│  - Concatenate [graph_embed || drug_embed]                 │
│  - 2-layer MLP → IC50                                      │
│                                                             │
│  TRAINING:                                                  │
│  - Multi-task: All drugs simultaneously                    │
│  - Loss: MSE (masked for missing values)                   │
│  - Your histology-based split for validation               │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### What to Compare Against

1. **Your baselines**: XGBoost, Random Forest with PCA (already done)
2. **MLP on same features**: Is graph structure helping, or just deep learning?
3. **GCN variant**: Is attention worth the complexity?

### Metrics to Track

| Metric | What It Tells You |
|--------|-------------------|
| Train MSE | Fitting capacity |
| Test MSE (histology split) | Generalization to unseen subtypes |
| Test MSE (site split) | Generalization to unseen tissues |
| Per-drug R² | Which drugs benefit most from the model |
| Attention entropy | Is the model focusing on few genes or spreading attention? |

---

## Appendix: Useful Libraries

| Library | Strengths | When to Use |
|---------|-----------|-------------|
| **PyTorch Geometric (PyG)** | Fast, extensive model zoo | Standard choice; great docs |
| **DGL (Deep Graph Library)** | Flexible, good for heterogeneous | Multi-relational graphs |
| **GraphNets (DeepMind)** | Clean abstractions | Research prototyping |
| **Spektral (Keras)** | Keras integration | If you prefer TF/Keras |

### PyG Example Skeleton

```python
import torch
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class DrugResponseGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_drugs):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        self.drug_embed = torch.nn.Embedding(num_drugs, 64)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    
    def forward(self, x, edge_index, batch, drug_idx):
        # GNN layers
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index)
        
        # Readout (pool nodes per graph)
        h_graph = global_mean_pool(h, batch)  # [batch_size, hidden_dim]
        
        # Drug embedding
        h_drug = self.drug_embed(drug_idx)    # [batch_size, 64]
        
        # Predict IC50
        out = self.predictor(torch.cat([h_graph, h_drug], dim=-1))
        return out.squeeze()
```

---

## Summary: Key Takeaways for Your Discussion

1. **GNNs learn node representations by aggregating neighbor information**—this lets you encode biological relationships (pathways, regulation) directly into the model.

2. **The main variants differ in how they aggregate**: GCN (normalized mean), GraphSAGE (sampled + flexible), GAT (learned attention), GIN (sum + MLP for expressivity).

3. **For drug response prediction**, you likely want:
   - Graph-level readout (one prediction per cell line)
   - Some form of drug conditioning (embedding or as graph node)
   - Biological prior graphs (PPI, pathways) as edges

4. **Start simple**: A 3-layer GAT on STRING PPI graph is a reasonable first model. You can add complexity (heterogeneous edges, learned structure, hierarchical pooling) based on what you learn.

5. **Your "harder" splits are the real test**—if GNNs help with unseen histologies/tissues, they're capturing transferable biological patterns beyond what flat feature vectors provide.

---

*Good luck with your architecture discussions! The goal of the first iteration should be to answer: "Does graph structure help at all?" Then iterate from there.*
