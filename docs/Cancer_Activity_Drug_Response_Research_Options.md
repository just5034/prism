# Connecting Drug Response Prediction to Cancer Activity Identification

**Document Created**: January 7, 2026
**Project**: Pharmacoepigenomics Multi-Task GNN
**Purpose**: Research options for linking drug sensitivity models to cancer biology

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Option 1: Multi-Task Learning (Cancer Subtype + Drug Response)](#option-1-multi-task-learning-cancer-subtype--drug-response)
3. [Option 2: Pathway Activity Scores as Cancer Signatures](#option-2-pathway-activity-scores-as-cancer-signatures)
4. [Option 3: Cancer Driver Gene Attention Mechanism](#option-3-cancer-driver-gene-attention-mechanism)
5. [Option 4: Oncogene Addiction Framework](#option-4-oncogene-addiction-framework)
6. [Option 5: TCGA Clinical Validation](#option-5-tcga-clinical-validation)
7. [Option 6: Contrastive Learning Between Cancer Subtypes](#option-6-contrastive-learning-between-cancer-subtypes)
8. [Comparison Summary](#comparison-summary)
9. [Recommended Reading](#recommended-reading)

---

## Problem Statement

### Current Setup
- **Data**: 987 cancer cell lines with DNA methylation (10K CpG sites) + gene expression (19K genes)
- **Target**: Drug response (IC50 values) for 265-375 compounds
- **Model**: Multi-task GNN for drug response prediction
- **Evaluation**: Random, histology-based, and site-based splits

### The Gap
The current approach predicts drug sensitivity but doesn't explicitly model **what makes these cells cancerous** or **why certain cancer features lead to drug sensitivity**.

### The Goal
Connect drug response prediction to cancer activity identification:
- What molecular features define "cancer-ness"?
- How do these features determine drug sensitivity?
- Can we identify cancer vulnerabilities through drug response modeling?

### Why Not Just Add Normal Cells?
Normal cell lines lack drug screening data (IC50 values), making direct integration into drug response models difficult. We need alternative approaches that leverage existing cancer biology knowledge.

---

## Option 1: Multi-Task Learning (Cancer Subtype + Drug Response)

### Scientific Background

**Multi-task learning (MTL)** is a machine learning paradigm where a model is trained on multiple related tasks simultaneously. The key insight is that tasks sharing underlying structure can benefit from joint training through **inductive transfer** - knowledge gained from one task improves performance on others.

In cancer biology, this is particularly relevant because:
- Cancer subtypes have distinct molecular profiles
- These molecular profiles influence drug sensitivity
- Learning both tasks simultaneously forces the model to capture biologically meaningful features

### The Biological Rationale

Different cancer types have different:
- **Driver mutations** (e.g., BRAF in melanoma, EGFR in lung adenocarcinoma)
- **Methylation patterns** (e.g., CIMP phenotype in colorectal cancer)
- **Expression programs** (e.g., EMT in metastatic cancers)
- **Drug sensitivities** (e.g., melanoma responds to BRAF inhibitors)

By jointly predicting cancer subtype AND drug response, the model must learn representations that capture these biological relationships.

### Mathematical Formulation

```
Given:
  X = input features (methylation + expression)
  Y_drug = drug response labels (IC50 values)
  Y_cancer = cancer subtype labels (histology)

Standard single-task:
  L = L_drug(f(X), Y_drug)

Multi-task:
  L = α * L_drug(f(X), Y_drug) + β * L_cancer(g(X), Y_cancer)

Where f and g share early layers (encoder)
```

### Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  [Methylation: 10K CpGs] + [Expression: 19K genes]     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 SHARED GNN ENCODER                      │
│  - Graph Attention layers (GAT)                        │
│  - Learns cancer-relevant molecular features           │
│  - Node embeddings capture gene-level information      │
│  - Graph pooling creates cell-line embedding           │
└─────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│    DRUG RESPONSE      │   │   CANCER SUBTYPE      │
│    PREDICTION HEAD    │   │   PREDICTION HEAD     │
│                       │   │                       │
│  - Drug embeddings    │   │  - Classification     │
│  - IC50 regression    │   │  - 56 histologies     │
│  - Multi-output       │   │  - Or 13 sites        │
└───────────────────────┘   └───────────────────────┘
            │                           │
            ▼                           ▼
       L_drug (MSE)              L_cancer (CrossEntropy)
            │                           │
            └───────────┬───────────────┘
                        ▼
                 L_total = α*L_drug + β*L_cancer
```

### Implementation Considerations

**Task weighting (α and β)**:
- Start with equal weights (α = β = 1.0)
- Use uncertainty-weighted losses (Kendall et al., 2018)
- Or learn weights dynamically during training

**Label granularity for cancer task**:
- Coarse: 13 primary sites (easier, less specific)
- Fine: 56 histologies (harder, more biologically specific)
- Hierarchical: Both levels simultaneously

**Potential issues**:
- **Negative transfer**: If tasks conflict, joint training can hurt both
- **Task dominance**: One task may dominate gradient updates
- **Class imbalance**: Some histologies have few samples

### What You Learn About Cancer

After training, the shared encoder captures features that are:
1. **Predictive of drug response** (by design)
2. **Discriminative of cancer type** (by design)
3. **Biologically meaningful** (by implication)

You can analyze:
- Which features are important for BOTH tasks (core cancer features)
- Which features are task-specific (drug-specific vs cancer-specific)
- How cancer subtypes cluster in the learned embedding space

### Key References

1. **Caruana, R. (1997)**. "Multitask Learning." Machine Learning, 28(1), 41-75.
   - Foundational paper on multi-task learning

2. **Kendall, A., Gal, Y., & Cipolla, R. (2018)**. "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR.
   - Principled approach to task weighting

3. **Rampasek, L., et al. (2019)**. "Dr.VAE: Drug Response Variational Autoencoder." Bioinformatics.
   - Multi-task drug response modeling

4. **Sharifi-Noghabi, H., et al. (2019)**. "MOLI: Multi-Omics Late Integration." Bioinformatics.
   - Multi-omics integration for drug response

### Pros and Cons

| Pros | Cons |
|------|------|
| Uses existing data | May not explicitly reveal cancer mechanisms |
| Straightforward implementation | Task weighting requires tuning |
| Proven approach in literature | Cancer classification may be "too easy" |
| Regularizes drug prediction | Shared encoder may be suboptimal for either task |

---

## Option 2: Pathway Activity Scores as Cancer Signatures

### Scientific Background

**Pathway analysis** moves beyond individual genes to analyze coordinated activity of gene sets representing biological processes. This is powerful because:

1. **Biological coherence**: Genes don't act alone; they function in pathways
2. **Dimensionality reduction**: Thousands of genes → hundreds of pathways
3. **Interpretability**: Pathway names are meaningful (e.g., "apoptosis", "cell cycle")
4. **Robustness**: Pathway scores are less noisy than individual gene measurements

**Gene Set Variation Analysis (GSVA)** and **single-sample GSEA (ssGSEA)** compute pathway activity scores for each sample, enabling comparison across samples.

### Cancer-Relevant Pathways

Cancer is fundamentally a disease of dysregulated pathways. The "Hallmarks of Cancer" (Hanahan & Weinberg, 2011) define core capabilities:

| Hallmark | Related Pathways | Drug Relevance |
|----------|------------------|----------------|
| Sustaining proliferative signaling | MAPK, PI3K/AKT, RTK signaling | MEK inhibitors, PI3K inhibitors |
| Evading growth suppressors | RB pathway, p53 pathway | CDK inhibitors |
| Resisting cell death | Apoptosis, autophagy | BCL2 inhibitors |
| Enabling replicative immortality | Telomerase, DNA repair | PARP inhibitors |
| Inducing angiogenesis | VEGF signaling, hypoxia | Anti-angiogenics |
| Activating invasion & metastasis | EMT, matrix remodeling | Experimental |
| Deregulating cellular energetics | Glycolysis, oxidative phosphorylation | Metabolic inhibitors |
| Genome instability | DNA damage response, MMR | PARP inhibitors, platinum drugs |

### Computing Pathway Activity Scores

**Method 1: GSVA (Gene Set Variation Analysis)**

```python
# R code (most common implementation)
library(GSVA)
library(msigdbr)

# Get hallmark gene sets
hallmarks <- msigdbr(species = "Homo sapiens", category = "H")

# Compute GSVA scores
gsva_scores <- gsva(
    expr = expression_matrix,  # genes x samples
    gset.idx.list = hallmark_gene_sets,
    method = "gsva"
)
# Output: pathways x samples matrix
```

**Method 2: ssGSEA (single-sample Gene Set Enrichment Analysis)**

```python
# Python with gseapy
import gseapy as gp

# For each sample
ssgsea_results = gp.ssgsea(
    data = expression_df,
    gene_sets = 'MSigDB_Hallmark_2020',
    outdir = None,
    no_plot = True
)
pathway_scores = ssgsea_results.res2d
```

**Method 3: Simple Mean/Median Aggregation**

```python
# Simplified approach
def compute_pathway_score(expression, pathway_genes):
    """Average expression of genes in pathway."""
    pathway_expr = expression[expression.index.isin(pathway_genes)]
    return pathway_expr.mean()

# For each pathway
pathway_scores = {}
for pathway_name, genes in pathway_dict.items():
    pathway_scores[pathway_name] = expression_df.apply(
        lambda x: compute_pathway_score(x, genes), axis=0
    )
```

### Integration with GNN

**Approach A: Pathway scores as graph-level features**

```
Cell line graph (nodes = genes)
        │
        ▼
   GNN encoder → graph embedding (128-dim)
        │
        ▼
   Concatenate with pathway scores (50-dim)
        │
        ▼
   [graph_embed || pathway_scores] → MLP → drug prediction
```

**Approach B: Pathway scores as node features**

```
For each gene node:
  features = [expression, methylation, pathway_membership_one_hot]

pathway_membership_one_hot = which pathways contain this gene
```

**Approach C: Pathway-level graph (meta-graph)**

```
Instead of gene-level graph:
  - Nodes = pathways (50-300 pathways)
  - Node features = pathway activity scores
  - Edges = pathway crosstalk (shared genes, known interactions)

This is similar to DRPreter architecture (see GNN research doc)
```

### Connecting to Cancer Activity

Pathway activity scores directly measure "cancer-ness":

| Observation | Interpretation |
|-------------|----------------|
| High cell cycle pathway activity | More proliferative, aggressive cancer |
| Low apoptosis pathway activity | Resistant to cell death |
| High EMT pathway activity | More invasive/metastatic |
| High DNA repair pathway activity | May resist DNA-damaging drugs |

**Analysis you can do**:
1. Correlate pathway scores with drug sensitivity
2. Identify which pathways predict response to specific drug classes
3. Cluster cell lines by pathway activity profiles
4. Compare to known cancer subtype signatures

### Methylation-Pathway Connection

Your data includes methylation, which regulates gene expression. You can:

1. **Compute "methylation pathway scores"**:
   - Average promoter methylation of genes in each pathway
   - High methylation → pathway silenced

2. **Compare expression vs methylation pathway scores**:
   - Identify pathways silenced by methylation
   - These represent epigenetically regulated cancer phenotypes

3. **Use both as features**:
   ```
   features = [expression_pathway_scores, methylation_pathway_scores]
   ```

### Key Resources

**Gene Set Databases**:
- **MSigDB** (Broad Institute): https://www.gsea-msigdb.org/gsea/msigdb/
  - Hallmark gene sets (50 sets, curated)
  - KEGG pathways
  - Reactome pathways
  - GO terms

- **Reactome**: https://reactome.org/
  - Curated pathway database
  - Hierarchical organization

**Software**:
- **GSVA** (R/Bioconductor): https://bioconductor.org/packages/GSVA/
- **gseapy** (Python): https://github.com/zqfang/GSEApy
- **ssGSEA2.0** (GenePattern): https://www.genepattern.org/

### Key References

1. **Hanahan, D., & Weinberg, R.A. (2011)**. "Hallmarks of Cancer: The Next Generation." Cell, 144(5), 646-674.
   - Foundational cancer biology framework

2. **Hanzelmann, S., Castelo, R., & Guinney, J. (2013)**. "GSVA: gene set variation analysis for microarray and RNA-Seq data." BMC Bioinformatics, 14, 7.
   - GSVA methodology paper

3. **Barbie, D.A., et al. (2009)**. "Systematic RNA interference reveals that oncogenic KRAS-driven cancers require TBK1." Nature, 462(7269), 108-112.
   - ssGSEA methodology paper

4. **Schubert, M., et al. (2018)**. "Perturbation-response genes reveal signaling footprints in cancer gene expression." Nature Communications, 9, 20.
   - PROGENy: pathway activity inference

### Pros and Cons

| Pros | Cons |
|------|------|
| Biologically interpretable | Pathway definitions may be incomplete |
| Reduces dimensionality | Loses gene-level resolution |
| Captures coordinated activity | Different methods give different results |
| Connects directly to cancer biology | Requires careful pathway selection |
| Many drug targets are pathway components | Pathway crosstalk complicates interpretation |

---

## Option 3: Cancer Driver Gene Attention Mechanism

### Scientific Background

**Cancer driver genes** are genes whose mutations or expression changes directly contribute to cancer development. Unlike "passenger" alterations (random, non-functional), driver alterations provide selective advantage to cancer cells.

Key driver gene databases:
- **COSMIC Cancer Gene Census**: ~730 genes with strong evidence
- **OncoKB**: Clinically annotated cancer genes
- **IntOGen**: Computationally identified drivers

**Attention mechanisms** in neural networks learn to weight different inputs based on their relevance to the prediction task. In graph neural networks, **Graph Attention Networks (GAT)** assign attention weights to edges (neighbor contributions).

### The Idea

Design your GNN to produce interpretable attention weights, then analyze:
1. **Which genes receive high attention?**
2. **Do high-attention genes overlap with known cancer drivers?**
3. **Does attention shift based on drug target?**

If your drug response model learns to attend to cancer driver genes, this validates that:
- The model captures cancer biology
- Drug sensitivity is linked to cancer mechanisms
- The model is not just memorizing patterns

### Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                    GENE-LEVEL GRAPH                     │
│                                                         │
│  Nodes: ~5,000 genes                                   │
│  Node features: [expression, methylation]              │
│  Edges: PPI network (STRING)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              GRAPH ATTENTION NETWORK (GAT)              │
│                                                         │
│  Layer 1: GATConv(in_dim, 128, heads=4)               │
│           → attention weights α_ij for each edge       │
│                                                         │
│  Layer 2: GATConv(512, 128, heads=4)                  │
│           → refined attention weights                  │
│                                                         │
│  Layer 3: GATConv(512, 128, heads=1)                  │
│           → final node embeddings                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              ATTENTION-WEIGHTED POOLING                 │
│                                                         │
│  Instead of mean pooling:                              │
│    graph_embed = Σ (attention_weight_i * node_embed_i) │
│                                                         │
│  attention_weight_i = softmax(MLP(node_embed_i))       │
│                                                         │
│  → Outputs: graph embedding + node attention weights   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 DRUG PREDICTION HEAD                    │
│                                                         │
│  [graph_embed || drug_embed] → MLP → IC50              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Extracting and Analyzing Attention

**Step 1: Extract attention weights after training**

```python
# PyTorch Geometric GAT returns attention weights
class InterpretableGAT(nn.Module):
    def forward(self, x, edge_index, return_attention=False):
        # Layer 1
        x, (edge_index1, attention1) = self.conv1(
            x, edge_index, return_attention_weights=True
        )

        # ... more layers ...

        # Attention pooling
        node_attention = self.attention_pool(x)  # [num_nodes, 1]
        graph_embed = (node_attention * x).sum(dim=0)

        if return_attention:
            return graph_embed, node_attention, attention1
        return graph_embed
```

**Step 2: Aggregate attention across samples**

```python
# For each gene, compute mean attention across all cell lines
gene_attention_scores = {}
for cell_line in cell_lines:
    _, node_attn, _ = model(cell_line_graph, return_attention=True)
    for gene, attn in zip(gene_names, node_attn):
        gene_attention_scores[gene].append(attn)

mean_attention = {g: np.mean(scores) for g, scores in gene_attention_scores.items()}
```

**Step 3: Compare to cancer driver genes**

```python
# Load COSMIC Cancer Gene Census
cosmic_drivers = load_cosmic_census()  # ~730 genes

# Statistical test: are drivers over-represented in high-attention genes?
top_attention_genes = get_top_k_genes(mean_attention, k=100)

overlap = set(top_attention_genes) & set(cosmic_drivers)
expected_overlap = (100 * len(cosmic_drivers)) / total_genes

# Hypergeometric test
from scipy.stats import hypergeom
p_value = hypergeom.sf(len(overlap), total_genes, len(cosmic_drivers), 100)

print(f"Observed overlap: {len(overlap)}")
print(f"Expected overlap: {expected_overlap:.1f}")
print(f"Enrichment p-value: {p_value:.2e}")
```

### Drug-Specific Attention Analysis

Different drugs should attend to different genes based on mechanism:

```python
# Compute attention for EGFR inhibitors vs MEK inhibitors
egfr_drugs = ['Erlotinib', 'Gefitinib', 'Lapatinib']
mek_drugs = ['Trametinib', 'Selumetinib', 'Cobimetinib']

egfr_attention = compute_mean_attention(model, cell_lines, egfr_drugs)
mek_attention = compute_mean_attention(model, cell_lines, mek_drugs)

# Check: do EGFR inhibitors attend to EGFR pathway genes?
egfr_pathway_genes = get_pathway_genes('EGFR_SIGNALING')
egfr_pathway_attention = np.mean([egfr_attention[g] for g in egfr_pathway_genes])

# Compare to random genes
random_attention = np.mean([egfr_attention[g] for g in random_genes])

print(f"EGFR pathway attention for EGFR drugs: {egfr_pathway_attention:.3f}")
print(f"Random gene attention: {random_attention:.3f}")
```

### Visualization

**Attention heatmap**:
```python
# Rows: cell lines (grouped by cancer type)
# Columns: top 50 high-attention genes
# Color: attention weight

import seaborn as sns
sns.clustermap(attention_matrix, row_colors=cancer_type_colors)
```

**Network visualization**:
```python
# Show gene network with node size = attention weight
# Highlight known cancer drivers in red
import networkx as nx

G = nx.from_edgelist(ppi_edges)
node_sizes = [mean_attention[g] * 100 for g in G.nodes()]
node_colors = ['red' if g in cosmic_drivers else 'blue' for g in G.nodes()]
nx.draw(G, node_size=node_sizes, node_color=node_colors)
```

### Key Resources

**Cancer Driver Gene Databases**:
- **COSMIC Cancer Gene Census**: https://cancer.sanger.ac.uk/census
- **OncoKB**: https://www.oncokb.org/
- **IntOGen**: https://www.intogen.org/
- **CIViC**: https://civicdb.org/

**Attention Mechanism Resources**:
- **PyTorch Geometric GAT**: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
- **Graph Attention Networks paper**: Velickovic et al. (2018), ICLR

### Key References

1. **Velickovic, P., et al. (2018)**. "Graph Attention Networks." ICLR.
   - Original GAT paper

2. **Sondka, Z., et al. (2018)**. "The COSMIC Cancer Gene Census: describing genetic dysfunction across all human cancers." Nature Reviews Cancer, 18, 696-705.
   - COSMIC Cancer Gene Census

3. **Bailey, M.H., et al. (2018)**. "Comprehensive Characterization of Cancer Driver Genes and Mutations." Cell, 173(2), 371-385.
   - Pan-cancer driver analysis (TCGA)

4. **Ying, R., et al. (2019)**. "GNNExplainer: Generating Explanations for Graph Neural Networks." NeurIPS.
   - Interpretability for GNNs

### Pros and Cons

| Pros | Cons |
|------|------|
| Highly interpretable | Attention ≠ importance (debated) |
| Validates biological relevance | Requires careful architecture design |
| Novel analysis angle | Post-hoc analysis, not guaranteed |
| Publication-worthy if drivers enriched | May not find enrichment if model is wrong |
| Drug-specific insights | Computationally intensive analysis |

---

## Option 4: Oncogene Addiction Framework

### Scientific Background

**Oncogene addiction** is the phenomenon where cancer cells become dependent on a single oncogene or pathway for survival and proliferation. Despite accumulating many mutations, cancer cells often remain critically dependent on one dominant driver.

Key examples:
| Cancer Type | Oncogene Addiction | Targeted Therapy |
|-------------|-------------------|------------------|
| CML | BCR-ABL fusion | Imatinib |
| BRAF-mutant melanoma | BRAF V600E | Vemurafenib |
| EGFR-mutant lung cancer | EGFR mutations | Erlotinib, Gefitinib |
| HER2+ breast cancer | HER2 amplification | Trastuzumab |
| ALK-fusion lung cancer | ALK rearrangements | Crizotinib |

**Synthetic lethality** is a related concept: two genes are synthetic lethal if loss of either alone is tolerable, but loss of both is lethal. This enables targeting "undruggable" oncogenes by inhibiting their synthetic lethal partners.

Key example:
- BRCA1/2-mutant cancers are sensitive to PARP inhibitors (synthetic lethality between BRCA and PARP)

### The Framework

Model drug sensitivity as a function of **oncogenic pathway activation**:

```
Hypothesis:
  Cancers "addicted" to pathway X → sensitive to drugs targeting pathway X

Formulation:
  Drug_sensitivity = f(pathway_activation, drug_target)

Where:
  - pathway_activation = computed from expression/methylation
  - drug_target = known target pathway of the drug
```

### Implementation Strategy

**Step 1: Define oncogenic pathway activation scores**

Using your expression data, compute activation of key oncogenic pathways:

```python
oncogenic_pathways = {
    'MAPK_ERK': ['BRAF', 'KRAS', 'NRAS', 'MEK1', 'MEK2', 'ERK1', 'ERK2'],
    'PI3K_AKT': ['PIK3CA', 'AKT1', 'AKT2', 'PTEN', 'MTOR'],
    'EGFR': ['EGFR', 'ERBB2', 'ERBB3'],
    'WNT': ['CTNNB1', 'APC', 'AXIN1', 'GSK3B'],
    'CELL_CYCLE': ['CCND1', 'CDK4', 'CDK6', 'RB1', 'CDKN2A'],
    'DNA_REPAIR': ['BRCA1', 'BRCA2', 'ATM', 'ATR', 'PARP1'],
    'APOPTOSIS': ['BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1'],
    # ... more pathways
}

def compute_pathway_activation(expression, pathway_genes, activators, repressors):
    """
    Compute pathway activation score.
    High expression of activators + low expression of repressors = high activation
    """
    activator_expr = expression[activators].mean()
    repressor_expr = expression[repressors].mean()
    return activator_expr - repressor_expr
```

**Step 2: Map drugs to target pathways**

You already have `PUTATIVE_TARGET` and `PATHWAY_NAME` in your GDSC data:

```python
# From your available_drugs.csv or GDSC metadata
drug_pathway_mapping = {
    'Trametinib': 'MAPK_ERK',
    'Selumetinib': 'MAPK_ERK',
    'Erlotinib': 'EGFR',
    'Gefitinib': 'EGFR',
    'Pictilisib': 'PI3K_AKT',
    'Olaparib': 'DNA_REPAIR',
    'Venetoclax': 'APOPTOSIS',
    # ... etc
}
```

**Step 3: Model the addiction relationship**

```python
# For each drug-cell line pair
for drug in drugs:
    target_pathway = drug_pathway_mapping[drug]

    for cell_line in cell_lines:
        pathway_activation = compute_pathway_activation(
            expression[cell_line],
            oncogenic_pathways[target_pathway]
        )

        # Hypothesis: high pathway activation → drug sensitivity
        # (if drug targets that pathway)

        # Add as feature
        features[cell_line, drug]['target_pathway_activation'] = pathway_activation
```

**Step 4: GNN architecture with pathway-drug interactions**

```
┌─────────────────────────────────────────────────────────┐
│                   CELL LINE ENCODER                     │
│                                                         │
│  Input: Gene expression + Methylation                   │
│  Output: pathway_activations (10-20 pathways)          │
│          cell_embedding (from GNN)                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    DRUG ENCODER                         │
│                                                         │
│  Input: Drug features (or learned embedding)           │
│  Include: target_pathway (one-hot or embedding)        │
│  Output: drug_embedding                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              PATHWAY-DRUG INTERACTION                   │
│                                                         │
│  For drug targeting pathway P:                         │
│    interaction = cell_pathway_activation[P] ⊙ drug_embed│
│                                                         │
│  This explicitly models:                               │
│    "If pathway P is active AND drug targets P,        │
│     predict high sensitivity"                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   PREDICTION HEAD                       │
│                                                         │
│  Input: [cell_embed || drug_embed || interaction]      │
│  Output: IC50 prediction                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### What This Reveals About Cancer

**Analysis 1: Pathway addiction profiles**

```python
# Cluster cell lines by pathway activation profiles
pathway_profiles = compute_all_pathway_activations(cell_lines)
# Cluster and visualize
# Compare to known cancer subtypes
```

**Analysis 2: Addiction-sensitivity correlation**

```python
# For each pathway-drug pair
for pathway, drugs in pathway_drug_pairs:
    activation = pathway_profiles[pathway]
    sensitivity = drug_response[drugs].mean(axis=1)

    correlation = pearsonr(activation, sensitivity)
    print(f"{pathway} addiction → {drugs} sensitivity: r={correlation:.3f}")
```

**Analysis 3: Identify novel addiction relationships**

```python
# Use trained model to predict:
# "If we increase pathway X activation, how does drug Y sensitivity change?"
# This reveals novel pathway-drug dependencies
```

### Connection to Precision Medicine

This framework directly models the logic of precision oncology:
1. **Diagnose**: Which pathway is this cancer addicted to?
2. **Treat**: Use drug targeting that pathway
3. **Predict**: Model predicts sensitivity based on addiction level

### Key References

1. **Weinstein, I.B. (2002)**. "Cancer: Addiction to Oncogenes." Science, 297(5578), 63-64.
   - Original oncogene addiction concept

2. **Torti, D., & Bhardwaj, N. (2011)**. "Oncogene addiction as a foundational rationale for targeted anti-cancer therapy." Cancer Research, 71(4), 1199-1205.
   - Review of oncogene addiction in therapy

3. **Iorio, F., et al. (2016)**. "A Landscape of Pharmacogenomic Interactions in Cancer." Cell, 166(3), 740-754.
   - Your data source! They analyze pathway-drug relationships

4. **Garnett, M.J., et al. (2012)**. "Systematic identification of genomic markers of drug sensitivity in cancer cells." Nature, 483(7391), 570-575.
   - GDSC paper, includes oncogene-drug analysis

### Pros and Cons

| Pros | Cons |
|------|------|
| Biologically grounded | Pathway definitions are imperfect |
| Directly actionable (precision medicine) | Addiction is pathway-specific, not universal |
| Interpretable predictions | Requires accurate drug-pathway mapping |
| Novel if applied to methylation | Some pathways lack good activation metrics |
| Connects to clinical practice | Ignores resistance mechanisms |

---

## Option 5: TCGA Clinical Validation

### Scientific Background

**The Cancer Genome Atlas (TCGA)** is the largest cancer genomics resource, containing multi-omic data (expression, methylation, mutations, copy number, clinical data) for 11,000+ tumors across 33 cancer types.

**The translational gap**: Cell lines are model systems, but do they reflect real tumors? Validating cell line findings in patient tumors is critical for clinical relevance.

### Available TCGA Data

| Data Type | Platform | Samples |
|-----------|----------|---------|
| Gene expression | RNA-seq (Illumina) | ~11,000 tumors |
| DNA methylation | Illumina 450K | ~10,000 tumors |
| Mutations | WES/WGS | ~10,000 tumors |
| Copy number | SNP arrays | ~11,000 tumors |
| Clinical | N/A | Varies by cancer type |

**Importantly**: TCGA includes **matched normal samples** for many tumors (~700 normal samples)

### Validation Strategy

```
┌─────────────────────────────────────────────────────────┐
│              PHASE 1: TRAIN ON CELL LINES               │
│                                                         │
│  Data: Your 987 cell lines (methylation + expression)  │
│  Task: Drug response prediction (IC50)                  │
│  Model: Multi-task GNN                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│             PHASE 2: APPLY TO TCGA TUMORS               │
│                                                         │
│  Data: TCGA tumors (methylation + expression)          │
│  Task: Predict drug sensitivity (no retraining!)       │
│  Output: Predicted IC50 for each TCGA tumor            │
│                                                         │
│  Note: Use same preprocessing pipeline                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│           PHASE 3: VALIDATE PREDICTIONS                 │
│                                                         │
│  A. Clinical outcome correlation:                       │
│     - Do predictions correlate with patient survival?  │
│     - Stratify by predicted sensitivity                │
│                                                         │
│  B. Treatment response (where available):              │
│     - Some TCGA tumors have treatment annotations      │
│     - Do predictions match actual response?            │
│                                                         │
│  C. Cancer subtype consistency:                        │
│     - Do predictions align with known subtype biology? │
│     - e.g., HER2+ breast → sensitive to HER2 drugs?   │
│                                                         │
│  D. Biomarker validation:                              │
│     - Do cell line biomarkers work in tumors?          │
│     - e.g., BRAF mutation → BRAF inhibitor sensitivity │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation Details

**Step 1: Download and preprocess TCGA data**

```python
# Option A: TCGAbiolinks (R)
library(TCGAbiolinks)

# Download methylation data
query_meth <- GDCquery(
    project = "TCGA-BRCA",  # or other cancer type
    data.category = "DNA Methylation",
    platform = "Illumina Human Methylation 450"
)
GDCdownload(query_meth)
data_meth <- GDCprepare(query_meth)

# Download expression data
query_expr <- GDCquery(
    project = "TCGA-BRCA",
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification"
)
GDCdownload(query_expr)
data_expr <- GDCprepare(query_expr)
```

```python
# Option B: Python with GDC API
import requests

# Query GDC API
files_endpt = "https://api.gdc.cancer.gov/files"
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "data_category", "value": "DNA Methylation"}},
        {"op": "=", "content": {"field": "platform", "value": "Illumina Human Methylation 450"}}
    ]
}
```

**Step 2: Align features between cell lines and TCGA**

```python
# Your cell line data uses 10K most variable CpGs
# TCGA has ~450K CpGs
# Must use SAME 10K CpGs for prediction

cell_line_cpgs = set(methylation_features)  # Your 10K CpGs
tcga_cpgs = set(tcga_methylation.columns)

common_cpgs = cell_line_cpgs & tcga_cpgs
print(f"Common CpGs: {len(common_cpgs)}")  # Should be ~10K

# Filter TCGA to common CpGs
tcga_methylation_filtered = tcga_methylation[list(common_cpgs)]
```

**Step 3: Apply cell line model to TCGA**

```python
# Load trained model
model = load_model('drug_response_gnn.pt')

# Predict for each TCGA tumor
tcga_predictions = {}
for tumor_id in tcga_samples:
    tumor_features = prepare_features(
        tcga_methylation_filtered.loc[tumor_id],
        tcga_expression.loc[tumor_id]
    )

    predictions = model.predict(tumor_features)  # IC50 for all drugs
    tcga_predictions[tumor_id] = predictions
```

**Step 4: Correlate with clinical outcomes**

```python
# Load clinical data
clinical = load_tcga_clinical('TCGA-BRCA')

# For a specific drug (e.g., Paclitaxel, commonly used in breast cancer)
drug = 'Paclitaxel'
predicted_sensitivity = [tcga_predictions[t][drug] for t in tcga_samples]

# Survival analysis
from lifelines import KaplanMeierFitter, CoxPHFitter

# Stratify by predicted sensitivity (median split)
high_sensitivity = predicted_sensitivity < np.median(predicted_sensitivity)

kmf = KaplanMeierFitter()
kmf.fit(clinical['survival_time'][high_sensitivity],
        clinical['event'][high_sensitivity],
        label='Predicted Sensitive')
kmf.plot()

kmf.fit(clinical['survival_time'][~high_sensitivity],
        clinical['event'][~high_sensitivity],
        label='Predicted Resistant')
kmf.plot()

# Log-rank test
from lifelines.statistics import logrank_test
result = logrank_test(
    clinical['survival_time'][high_sensitivity],
    clinical['survival_time'][~high_sensitivity],
    clinical['event'][high_sensitivity],
    clinical['event'][~high_sensitivity]
)
print(f"Log-rank p-value: {result.p_value:.4f}")
```

### What This Proves

| Finding | Interpretation |
|---------|----------------|
| Predictions correlate with survival | Model captures clinically relevant biology |
| Predictions match treatment response | Model is predictive, not just descriptive |
| Predictions align with known biomarkers | Model learns established cancer biology |
| Predictions don't correlate | Cell line model doesn't transfer (negative result, still informative) |

### Challenges

1. **Feature alignment**: Cell line and TCGA data may use different preprocessing
2. **Batch effects**: Different platforms, labs, collection methods
3. **Biological differences**: Cell lines ≠ tumors (tumor microenvironment, heterogeneity)
4. **Limited treatment data**: Most TCGA patients received standard of care, not the drugs you're predicting

### Key Resources

**Data Access**:
- **GDC Portal**: https://portal.gdc.cancer.gov/
- **Xena Browser**: https://xenabrowser.net/ (easier interface)
- **cBioPortal**: https://www.cbioportal.org/ (for exploration)

**Software**:
- **TCGAbiolinks** (R): https://bioconductor.org/packages/TCGAbiolinks/
- **GDC Client**: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

### Key References

1. **Cancer Genome Atlas Research Network (2013-2018)**. Multiple papers across 33 cancer types.
   - Pan-Cancer Atlas papers in Cell (2018)

2. **Geeleher, P., et al. (2014)**. "Clinical drug response can be predicted using baseline gene expression levels and in vitro drug sensitivity in cell lines." Genome Biology, 15, R47.
   - Cell line → patient prediction validation

3. **Iorio, F., et al. (2016)**. "A Landscape of Pharmacogenomic Interactions in Cancer." Cell.
   - Includes TCGA comparison of cell line findings

4. **Rees, M.G., et al. (2016)**. "Correlating chemical sensitivity and basal gene expression reveals mechanism of action." Nature Chemical Biology, 12, 109-116.
   - CTRP dataset, includes validation approaches

### Pros and Cons

| Pros | Cons |
|------|------|
| Gold standard for clinical relevance | Large data download (~TB) |
| Includes survival outcomes | Batch effects are real |
| Multiple cancer types | Limited treatment response data |
| Matched normal samples available | Cell line → patient gap |
| Strengthens publication impact | Adds significant scope to project |

---

## Option 6: Contrastive Learning Between Cancer Subtypes

### Scientific Background

**Contrastive learning** is a self-supervised learning paradigm where models learn representations by contrasting positive pairs (similar samples) against negative pairs (dissimilar samples). The key idea: learn embeddings where similar samples are close and dissimilar samples are far apart.

**SimCLR**, **MoCo**, and **BYOL** are popular contrastive learning frameworks in computer vision. In biology, contrastive learning has been applied to:
- Single-cell data (learning cell type representations)
- Drug discovery (learning molecular representations)
- Protein structure (learning sequence embeddings)

### Application to Cancer Drug Response

Instead of cancer vs. normal contrast, use **cancer subtype contrasts**:

```
Positive pairs (pull together):
  - Same histology, similar drug response profile
  - Same primary site
  - Same molecular subtype

Negative pairs (push apart):
  - Different histology, different drug response
  - Different primary site
  - Different molecular characteristics
```

### The Intuition

By learning to distinguish cancer subtypes based on their molecular features, the model:
1. Captures what makes each cancer type unique (cancer biology)
2. Learns representations useful for downstream drug prediction
3. Provides interpretable embeddings (similar cancers cluster together)

### Architecture Design

```
┌─────────────────────────────────────────────────────────┐
│                  CONTRASTIVE FRAMEWORK                  │
└─────────────────────────────────────────────────────────┘

Step 1: Augmentation (create positive pairs)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  For cell line X:                                       │
│    X' = augment(X)  # Augmented version                │
│                                                         │
│  Augmentation strategies:                               │
│    - Feature dropout (randomly mask CpGs/genes)        │
│    - Gaussian noise addition                            │
│    - Feature permutation within pathways               │
│    - Subsampling features                              │
│                                                         │
│  (X, X') is a positive pair                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 2: Encoder (shared weights)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  z = GNN_encoder(X)    # Embedding of original         │
│  z' = GNN_encoder(X')  # Embedding of augmented        │
│                                                         │
│  Same encoder, shared weights                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 3: Contrastive Loss
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  NT-Xent Loss (Normalized Temperature-scaled):         │
│                                                         │
│  L = -log[exp(sim(z,z')/τ) / Σ exp(sim(z,z_k)/τ)]     │
│                                                         │
│  Where:                                                 │
│    sim(a,b) = cosine similarity                        │
│    τ = temperature hyperparameter                       │
│    z_k = embeddings of other samples (negatives)       │
│                                                         │
│  This encourages:                                       │
│    - (z, z') to be similar (positive pair)             │
│    - (z, z_k) to be dissimilar (negative pairs)        │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 4: Cancer-Aware Contrastive (Novel!)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Beyond just (X, X') as positives:                     │
│                                                         │
│  Hard positives:                                        │
│    - Same histology cell lines                         │
│    - Same primary site cell lines                      │
│                                                         │
│  Hard negatives:                                        │
│    - Different histology cell lines                    │
│    - Different primary site cell lines                 │
│                                                         │
│  This forces the model to learn cancer subtype         │
│  distinctions!                                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
Step 5: Fine-tuning for Drug Prediction
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  After contrastive pre-training:                       │
│                                                         │
│  Freeze encoder OR continue training                    │
│  Add drug prediction head                              │
│  Fine-tune on IC50 prediction                          │
│                                                         │
│  The encoder now has cancer-aware representations!     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Implementation Pseudocode

```python
import torch
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels=None):
        """
        z1, z2: embeddings of positive pairs [batch_size, embed_dim]
        labels: cancer subtype labels (optional, for supervised contrastive)
        """
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate for similarity computation
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, embed_dim]

        # Compute pairwise similarity
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -float('inf'))

        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        pos_mask = torch.zeros_like(sim).bool()
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True

        # If using labels, same-class samples are also positives
        if labels is not None:
            labels = torch.cat([labels, labels])
            for i in range(2 * batch_size):
                for j in range(2 * batch_size):
                    if i != j and labels[i] == labels[j]:
                        pos_mask[i, j] = True

        # NT-Xent loss
        pos_sim = sim[pos_mask].view(2 * batch_size, -1)
        neg_sim = sim[~pos_mask & ~mask].view(2 * batch_size, -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class CancerContrastiveModel(nn.Module):
    def __init__(self, gnn_encoder, projection_dim=128):
        super().__init__()
        self.encoder = gnn_encoder
        self.projector = nn.Sequential(
            nn.Linear(gnn_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x, edge_index, batch):
        # Get graph embeddings
        h = self.encoder(x, edge_index, batch)
        # Project to contrastive space
        z = self.projector(h)
        return z

    def get_embeddings(self, x, edge_index, batch):
        """Get embeddings without projection (for downstream tasks)."""
        return self.encoder(x, edge_index, batch)
```

### What This Learns About Cancer

After contrastive pre-training, analyze the learned representations:

```python
# Extract embeddings for all cell lines
embeddings = []
labels = []
for cell_line, subtype in cell_lines:
    emb = model.get_embeddings(cell_line)
    embeddings.append(emb)
    labels.append(subtype)

embeddings = np.stack(embeddings)

# Visualize with UMAP/t-SNE
from umap import UMAP
umap = UMAP(n_neighbors=15, min_dist=0.1)
emb_2d = umap.fit_transform(embeddings)

plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20')
plt.title("Cancer Subtypes in Learned Embedding Space")
```

**Expected result**: Cell lines cluster by cancer subtype, showing the model learned cancer biology.

### Novel Aspects

1. **Cancer-supervised contrastive loss**: Using cancer subtype labels to define positive/negative pairs
2. **Multi-view contrast**: Methylation view vs. expression view as two "augmentations"
3. **Drug response-informed contrast**: Cell lines with similar drug profiles are positives

### Key References

1. **Chen, T., et al. (2020)**. "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)." ICML.
   - Foundational contrastive learning paper

2. **Khosla, P., et al. (2020)**. "Supervised Contrastive Learning." NeurIPS.
   - Using labels in contrastive learning

3. **You, Y., et al. (2020)**. "Graph Contrastive Learning with Augmentations." NeurIPS.
   - Contrastive learning for graphs

4. **Wang, H., et al. (2021)**. "Self-Supervised Contrastive Learning for Integrative Single Cell RNA-seq Data Analysis." bioRxiv.
   - Contrastive learning in biology

### Pros and Cons

| Pros | Cons |
|------|------|
| Self-supervised (leverages unlabeled data) | Complex implementation |
| Learns transferable representations | Hyperparameter sensitive (temperature, augmentations) |
| Novel application to cancer | May not outperform supervised approach |
| Provides interpretable embeddings | Requires large batch sizes |
| Trendy in ML community | Augmentation design is non-trivial for biological data |

---

## Comparison Summary

| Option | Novelty | Difficulty | Data Needed | Impact | Time Estimate |
|--------|---------|------------|-------------|--------|---------------|
| 1. Multi-Task (subtype + drug) | Low | Easy | Existing | Medium | 1-2 weeks |
| 2. Pathway Activity Scores | Medium | Easy | MSigDB (free) | Medium | 1-2 weeks |
| 3. Cancer Driver Attention | High | Medium | COSMIC (free) | High | 2-3 weeks |
| 4. Oncogene Addiction | High | Medium | Existing | High | 2-3 weeks |
| 5. TCGA Validation | Medium | Hard | TCGA (~100GB) | Very High | 3-4 weeks |
| 6. Contrastive Learning | High | Hard | Existing | High | 3-4 weeks |

### Recommended Combinations

**Conservative Path** (Lower risk, solid publication):
- Option 1 (Multi-Task) + Option 2 (Pathways) + Option 5 (TCGA)
- Strong foundation, clinical validation, interpretable

**Innovative Path** (Higher risk, novel contribution):
- Option 3 (Driver Attention) + Option 4 (Oncogene Addiction) + Option 5 (TCGA)
- Novel framework, biological insight, clinical validation

**ML-Focused Path** (Methods contribution):
- Option 6 (Contrastive) + Option 3 (Driver Attention)
- Novel methods, interpretability, potentially Nature Methods-level

---

## Recommended Reading

### Cancer Biology
1. Hanahan & Weinberg (2011). "Hallmarks of Cancer: The Next Generation." Cell.
2. Vogelstein et al. (2013). "Cancer Genome Landscapes." Science.
3. Bailey et al. (2018). "Comprehensive Characterization of Cancer Driver Genes." Cell.

### Drug Response Prediction
4. Iorio et al. (2016). "A Landscape of Pharmacogenomic Interactions in Cancer." Cell.
5. Geeleher et al. (2014). "Clinical drug response can be predicted using baseline gene expression." Genome Biology.
6. Costello et al. (2014). "A community effort to assess and improve drug sensitivity prediction algorithms." Nature Biotechnology.

### Graph Neural Networks in Biology
7. Zitnik et al. (2018). "Modeling polypharmacy side effects with graph convolutional networks." Bioinformatics.
8. Fout et al. (2017). "Protein Interface Prediction using Graph Convolutional Networks." NeurIPS.
9. Your existing: `GNN_Research_Compilation.md`

### Multi-Task Learning
10. Caruana (1997). "Multitask Learning." Machine Learning.
11. Kendall et al. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR.

### Contrastive Learning
12. Chen et al. (2020). "SimCLR: A Simple Framework for Contrastive Learning." ICML.
13. You et al. (2020). "Graph Contrastive Learning with Augmentations." NeurIPS.

---

## Next Steps

1. **Discuss with your group**: Which direction aligns with your goals and timeline?
2. **Start simple**: Options 1-2 can be implemented quickly to establish baselines
3. **Build iteratively**: Add complexity (Options 3-4) based on initial results
4. **Plan for validation**: Budget time for TCGA integration (Option 5)
5. **Consider novelty**: If aiming for top venues, Options 3, 4, or 6 offer more novelty

---

*Document prepared for research planning. Revisit and update as the project progresses.*
