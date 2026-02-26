# Project Scope & Team Pitch: Precision Oncology Pipeline

## The Elevator Pitch

We're building a system that takes a cancer patient's tumor data and predicts **which drugs will work for that specific tumor**. We do this by standing on the shoulders of existing foundation models -- large AI models that already understand biology -- and training a small, focused prediction layer on top for drug response.

Our collaborator's protein modeling team extends this further: our predictions feed into their pipeline to identify **new drug targets** (specifically for Antibody-Drug Conjugates).

---

## Our Team's Core Task

**We predict drug response. That hasn't changed.**

```
INPUT:   A tumor's molecular profile
         - DNA methylation (which genes are silenced/activated)
         - Gene expression (which genes are producing proteins)

OUTPUT:  Predicted drug response
         - For each available drug: will it work on THIS tumor?
         - Ranked list of drugs most likely to be effective
         - Confidence scores
```

Our training labels are IC50 drug response values from 925 cancer cell lines (GDSC database). That is our ground truth. That stays the same.

---

## The Technical Approach: Foundation Models + Task-Specific Heads

### The Old Plan (harder, riskier)

Build a GNN from scratch. Train it on our 925 cell lines to learn BOTH biology (what methylation patterns mean) AND pharmacology (how that relates to drug response). This is asking a model to learn everything from limited data. Our baselines failed on generalization splits (negative R-squared) because 925 samples isn't enough.

### The New Plan (collaborator's suggestion, smarter)

Use **pre-trained foundation models** that already understand biology. We only train a small "head" on top for drug response prediction.

```
FOUNDATION MODELS (already trained, we use as-is)
=================================================
These exist. We don't build them. We use their outputs.

  Protein language model (SaprotHub)     --> understands protein structure/function
  Gene expression model (Geneformer etc) --> understands gene regulatory patterns
  Drug encoder (pre-trained GIN/ChemBERTa) --> understands molecular structure

                    |
                    | Each model converts raw data into rich "embeddings"
                    | (numerical vectors that encode deep biological meaning)
                    |
                    v

OUR WORK: THE PREDICTION HEAD
=================================================
This is what we build and train.

  Tumor embedding (from foundation models above)
  + Drug embedding (from drug encoder above)
  |
  v
  Small neural network (few layers, ~100K-1M parameters)
  |
  v
  Predicted IC50 drug response

  Training data: 925 cell lines with known IC50 values
  This is fast (minutes, not hours) and 925 samples is plenty for a small head.
```

**Analogy**: Instead of teaching someone English from birth so they can translate a medical document, you hire someone who already speaks English (foundation model) and just teach them medical terminology (the head). Much faster, much more reliable.

### Why this works better

| | Build from scratch | Foundation model + head |
|---|---|---|
| Parameters to train | ~10-50 million | ~100K-1M |
| Data needed | Thousands+ | Hundreds is fine |
| Training time | Hours on GPU | Minutes |
| Biology knowledge | Must learn from our data | Already captured |
| Iteration speed | Slow | Fast (swap heads, try variants) |
| Risk | High | Low (proven approach) |
| Generalization | Poor (negative R-squared) | Expected to be much better |

---

## Where TCGA Fits In

TCGA (The Cancer Genome Atlas) = ~11,000 real patient tumors across 33 cancer types. It has methylation, expression, mutations, clinical data. **But no IC50 drug response values.**

So what's it for?

### 1. Validating that our embeddings are good
Before training the drug response head, we check: do TCGA tumors cluster by cancer type in the foundation model's embedding space? If yes, the embeddings capture real biology. If not, we need a different foundation model.

### 2. Sanity-checking our drug predictions
~20-30 GDSC drugs overlap with drugs actually given to TCGA patients. TCGA records whether patients responded. We can check: "our model predicts these tumors are cisplatin-sensitive -- were those the patients who actually responded?" This is validation, not training.

### 3. Adopting the standard data format
TCGA's schema is the universal standard in cancer genomics. If our pipeline reads TCGA data, it's compatible with every other oncology dataset and every collaborator's tools. This is what our collaborator specifically asked for.

### 4. Optional: additional fine-tuning signal
Where TCGA has coarse drug response labels (responded / didn't respond), we can use those as weak supervision to supplement our 925 IC50 training samples. Not the primary use, but it helps.

**Bottom line**: TCGA doesn't give us more drug response training data. It gives us a standard data format, a validation path to real clinical outcomes, and confidence that our approach captures real biology.

---

## The Full Picture: Our Team + Collaborator

```
OUR TEAM                               COLLABORATOR'S TEAM
--------                               -------------------

Tumor data (TCGA schema)
    |
    v
Foundation model embeddings
(expression, methylation)
    |                                  Protein language models
    v                                  (SaprotHub, fine-tuned)
Drug response head                              |
    |                                           v
    v                                  Protein druggability scores
DRUG RESPONSE PREDICTIONS  ----------> ADC target discovery
(our output)                           (their application)
```

- We hand them: "these genes are overexpressed in these tumors, and these drugs are predicted to work"
- They determine: "can we design new ADCs against these protein targets?"
- Two teams, two outputs, shared data format (TCGA schema), shared compute (HPC)

---

## Who Does What (Team Roles)

### Our Team (Genomics / ML side)

#### Person A -- Data Engineering & TCGA Integration
- Download and process TCGA data (~11K patients, 33 cancer types)
- Reformat our existing pipeline to use TCGA's schema
- Set up foundation model inference pipeline (run data through pre-trained models, store embeddings)
- Maintain the HPC environment (packages, data storage, job scripts)

**Day-to-day**: Python/pandas, large data wrangling (HDF5/parquet), HPC batch jobs, data pipeline code. Needs comfort with data at scale (~50-200 GB).

#### Person B -- Model Development (Prediction Heads)
- Design and train the drug response prediction head
- Experiment with different head architectures (MLP, small GNN, attention)
- Hyperparameter tuning, ablation studies
- Integrate protein embeddings from collaborator when available

**Day-to-day**: PyTorch code, GPU training runs, experiment tracking (W&B or similar), reading papers. Most ML-heavy role but lighter than building from scratch.

#### Person C -- Analysis & Biology Interpretation
- Validate that foundation model embeddings make biological sense
- Interpret model predictions: which features drive drug response?
- Compare predictions against known cancer biology
- Generate figures, write up results

**Day-to-day**: Jupyter notebooks, statistical analysis, visualization, literature review. Good role for someone stronger in biology than engineering.

#### Project Lead -- Architecture & Coordination
- Design how foundation models, heads, and data pipeline connect
- Interface with collaborator's protein team
- Present to Jaco / Dell on compute needs
- Ensure pipeline flows end-to-end

### Collaborator's Team (Protein Modeling side)

**What they provide:**
- Protein language model expertise (SaprotHub fine-tuning)
- Protein druggability scoring for candidate targets
- Domain knowledge on ADC target selection

**What they need from us:**
- Gene expression predictions in TCGA format
- List of candidate overexpressed proteins per cancer type
- Shared HPC access

---

## The Concrete Workstreams

### Phase 1: Foundation (Weeks 1-3)

*Get set up, data flowing, models accessible*

| Task | Who | What it looks like |
|---|---|---|
| Get team on HPC | Lead + Person A | Account setup, SSH, install PyTorch, test GPU |
| Download TCGA data | Person A | GDC Data Transfer Tool, ~200 GB, organize by cancer type |
| Adopt TCGA schema | Person A + Lead | Rewrite data loaders for standard column names, gene IDs, CpG IDs |
| Set up foundation models | Person B + Lead | Download pre-trained models (Geneformer, SaprotHub, drug encoders), test inference |
| Prep Jaco presentation | Lead | "Here's the data, the compute need, and the science" |
| Review SaprotHub paper | Lead + Person B | Understand protein LM fine-tuning, ColabSaprot interface |

**Deliverable**: Everyone on HPC, data loadable, foundation models producing embeddings.

### Phase 2: Build the Core (Weeks 3-8)

*Generate embeddings, train drug response head*

| Task | Who | What it looks like |
|---|---|---|
| Generate tumor embeddings | Person A + B | Run all cell lines + TCGA patients through foundation models, store embeddings |
| Train drug response head | Person B | Small head on top of embeddings, train on 925 cell lines with IC50 |
| Validate embeddings on TCGA | Person C | Do cancer types cluster? Do known biology patterns appear? |
| Protein target scoring | Collaborator | Score candidate proteins for druggability |
| Baseline comparison | Person C | Compare head performance vs our old from-scratch models |

**Deliverable**: Working drug response predictor. Validation that it beats our old baselines, especially on hard generalization splits.

### Phase 3: Integrate & Validate (Weeks 8-12)

*Add protein features, validate clinically, discover targets*

| Task | Who | What it looks like |
|---|---|---|
| Add protein embeddings to head | Person B + Collaborator | Concatenate protein features from SaprotHub into prediction input |
| Clinical validation on TCGA | Person A + C | Check predictions against real patient outcomes for overlapping drugs |
| ADC target discovery | Lead + Collaborator | Identify new targets: overexpressed + druggable + predicted responsive |
| Write up results | Everyone | Paper draft, figures, supplementary data |

**Deliverable**: End-to-end pipeline. Clinical validation. Novel ADC target candidates. Paper-ready results.

---

## What to Tell Jaco (The Compute Pitch)

Frame for an HPC specialist:

### Data Scale
- **TCGA**: ~11,000 patients x 33 cancer types x multi-omics (~200-500 GB)
- **Foundation model inference**: Running 12K+ samples through large pre-trained models
- **Embeddings storage**: ~10-50 GB of pre-computed embeddings

### Compute Needs
- **Foundation model inference**: GPU-bound, large batch sizes, one-time cost per dataset
- **Head training**: Light GPU work, but hundreds of experiments (hyperparameter search)
- **Protein LM fine-tuning**: Collaborator's workload, similar GPU needs
- **Data preprocessing**: CPU-bound, high memory (64GB+ RAM), I/O heavy

### Why HPC, Not Cloud
- Iterative research -- hundreds of experiments, not one big job
- Data locality matters at 500GB+ scale
- Multi-team collaboration needs shared storage and compute
- Foundation model weights are large (multi-GB), better to keep local

---

## The One-Liner for Each Audience

**For teammates**: "We use existing AI models that already understand biology, and we train a small drug response predictor on top. Your piece is [specific role]."

**For Jaco**: "We need your machines to run 11,000 patient genomes through foundation models and train drug response predictors. Two teams, shared infrastructure."

**For our collaborator**: "We produce drug response predictions from foundation model embeddings. You score protein druggability. TCGA schema is our common language."

**For a paper abstract**: "We leverage pre-trained biological foundation models with task-specific prediction heads to predict cancer drug response, validated on TCGA clinical outcomes, and identify novel ADC targets through integrated protein structural analysis."

---

## Key Background Concepts

### TCGA (The Cancer Genome Atlas)
NIH-funded dataset of ~11,000 primary patient tumors across 33 cancer types. Contains matched multi-omics data per patient: DNA methylation, gene expression, mutations, copy number, protein expression, and clinical outcomes. The standard schema in cancer genomics. **No IC50 drug response values** -- has coarse clinical response instead.

### ADCs (Antibody-Drug Conjugates)
Cancer drugs with three parts: an antibody (finds the tumor), a linker, and a toxic payload (kills the cell). Only work if there's a good protein target on the tumor surface. ~15 FDA-approved as of 2025. The bottleneck is finding new targetable proteins. Recent computational work has expanded candidate targets from ~15 to 75+.

### Foundation Models (the key concept)
Large AI models pre-trained on massive datasets that learn general representations. Like how GPT learned language from the internet, biological foundation models learned biology from millions of protein sequences, gene expression profiles, etc. We use their "understanding" as features and just train a small task-specific layer on top. This is dramatically more efficient than training from scratch.

### SaprotHub
A collaborative platform (Nature Biotechnology, 2025) that lets researchers fine-tune protein language models without deep ML expertise. Our collaborator's team uses this to score proteins for druggability, surface accessibility, and binding properties.

### Prediction Head
A small neural network (few layers) added on top of foundation model outputs. It takes the rich embeddings from the foundation model and maps them to our specific prediction target (IC50 drug response). This is the only part we train -- everything else is pre-trained.

---

## What We Have Already

| Asset | Details |
|---|---|
| GSE68379 dataset | 1,028 cancer cell lines, 485K CpG sites, fully processed |
| GSE270494 dataset | 180 hematological malignancy cell lines, 760K CpG sites |
| ML-ready datasets | Methylation + drug response (925 lines, 375 drugs) + gene expression (987 lines, 19K genes) |
| Baseline models | Random Forest, XGBoost -- tissue classification works, drug response on hard splits fails |
| GNN architecture proposals | 3 architectures documented (may pivot to foundation model + head approach) |
| EDA & visualization | Complete for both datasets |
| Documentation | Session history, coding standards, metadata |

## What We Need

| Need | Source | Purpose |
|---|---|---|
| TCGA multi-omics data | GDC Data Portal | Standard schema, validation, embedding quality checks |
| Foundation models | Geneformer, SaprotHub, ChemBERTa | Pre-trained biology/chemistry understanding |
| HPC access | Jaco / Dell partnership | Run inference and training at scale |
| Protein modeling expertise | Collaborator's team | Druggability scoring, ADC target identification |
| TCGA clinical annotations | GDC (controlled access, needs dbGaP) | Drug response validation |

---

## FAQ for Teammates

**Q: Are we still predicting drug response?**
A: Yes. Our core output is: given a tumor profile, predict which drugs will work. The approach changes (foundation models instead of from-scratch), but the goal is the same.

**Q: Why don't we just use TCGA patients for training?**
A: TCGA has no IC50 values. Our training labels (drug response) still come from the 925 GDSC cell lines. TCGA is for validation and for teaching foundation models about tumor biology.

**Q: What's the collaborator's team actually doing?**
A: They take our gene expression predictions and score the resulting proteins for druggability. Their output is "can we design an ADC against this target?" They don't do drug response prediction -- we do.

**Q: Why foundation models instead of building our own GNN?**
A: 925 samples is too few to learn biology AND pharmacology from scratch (our baselines prove this -- negative R-squared on hard splits). Foundation models already know the biology. We just teach a small head the pharmacology part.

**Q: What does TCGA schema adoption actually mean for my code?**
A: Renaming columns, using standard gene identifiers (Ensembl IDs), standard CpG probe naming, organizing by cancer type. It means our pipeline can ingest any TCGA-formatted dataset and any collaborator can use our outputs.

---

## References

- SaprotHub paper: https://www.nature.com/articles/s41587-025-02859-7
- AI in ADC development: https://www.nature.com/articles/s41698-025-01159-2
- Expanding ADC targets: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0308604
- ADC target atlas: https://www.nature.com/articles/s41417-023-00701-3
- GNN Architecture Proposals: docs/GNN_Architecture_Proposals.md
- Geneformer: https://www.nature.com/articles/s41586-023-06139-9
