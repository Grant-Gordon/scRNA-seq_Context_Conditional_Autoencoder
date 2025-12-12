# Context-Conditioned Cycle Autoencoder for scRNA-seq

## Overview

This repository implements a **cycle-consistency autoencoder (CAE)** for single-cell RNA-sequencing (scRNA-seq) data that explicitly conditions gene expression on multiple categorical metadata fields (e.g. assay, cell type, sex).

The core goal of the project is to build an **interpretable, additive model** of how different metadata contexts contribute to gene expression, while also enabling **counterfactual generation** (e.g. transforming a cell from one biological context to another).

---

## High-level Model Description

At a high level, the model:

* Takes sparse gene-expression matrices (`.npz`) as input
* Uses aligned per-cell metadata (`.pkl`) as conditioning context
* Learns a shared latent representation of gene expression
* Applies metadata-specific transformations via context-dependent heads
* Trains using cycle consistency and a classifier-based conditioning signal
* Produces **additive, interpretable contributions** from each metadata field

---

## Repository Structure

```text
src/
├── dataloader.py              # Chunk-level and batch-level datasets
├── trainer.py                 # Training loop and optimizer logic
├── logging_helpers.py         # TensorBoard + logging utilities
├── models/
│   ├── CAE.py                 # Core model architecture
│   └── context_classifier.py  # Metadata classifier
├── scripts/
│   ├── main.py                # Primary Python entrypoint
│   ├── metadata_preprocessor.py
│   └── slurm_submission.sh    # SLURM job script (HPC entrypoint)
```

Typical run-time–generated directories (not versioned):

* `job-outputs/` – training outputs, logs, TensorBoard files
* `preprocessed_metadata/` – metadata vocabularies and field specifications

---

## Data Format and Loading Model

### Input data

* Gene expression: sparse CSR matrices (`.npz`)
* Metadata: per-cell `.pkl` files aligned with expression rows
* Data are chunked at the file level to support large datasets

### Two-level DataLoader design

Training uses **two nested PyTorch DataLoaders**:

#### 1. Chunk loader (outer loader)

* Iterates over `(expression_chunk, metadata_chunk)` pairs
* Soley responsible for preloading chunks from disk
* Utilizes pytorch dataloader's `prefetch_factor` & `num_workers` to preload in background
* Batch size=1 intentionally as this dataloader should ONLY be loading from disk

#### 2. Within-chunk loader (inner loader)

* Creates minibatches of cells from a single chunk
* Performs the actual batching seen by the model

It is reccomended to always have a chunk preloaded if your RAM permits as chunk I/O significantly increases wall-time and GPU idle time.

---

## Model Architecture

### Base autoencoder head

The base component is a **single-layer, tied-weights linear autoencoder**:

* Encoder: `h_base = X Wᵀ`
* Decoder: `X̂_base = h_base W`

Key properties:

* No biases
* Weights constrained to be non-negative
* Trained on every batch, regardless of metadata

This head captures shared structure in gene expression that is independent of metadata context.

---

### Metadata conditioning heads

For each enabled metadata field (e.g. `assay`, `cell_type`, `sex`), the model instantiates a conditioning head consisting of:

* A **field-shared encoder** (`input_dim → latent_dim`)
* A **field-shared decoder** (`latent_dim → input_dim`)
* A pool of **context-specific FFNs** (`latent_dim → latent_dim`), one per category value

Note:
- Unlike the Base-encoder Shared encoders and decoders are *not* tied
- Context-specific FFNs are only applied to samples that match that context   
e.g. a cell with context liver sees: `Enc_cell_type -> liver_FFN -> Dec_cell_type`  
while a cell with context heart sees: `Enc_cell_type -> heart_FFN -> Dec_cell_type`

#### Context grouping logic

During a forward pass:

1. Samples are grouped by context value
2. Only the relevant FFN heads are applied
3. Outputs are reassembled into a full batch

This keeps computation efficient even for high-cardinality metadata fields.

---

### Output composition and scaling

The final generated expression is the **sum of all head outputs**:

```
X_st = X̂_base + Σ_f X̂_f
```

Exploratory runs showed metadata heads dominating training with the base head contributing negligably; this caused the model to fail to converge.  To stabilize training:

* Metadata decoder logits are scaled by a fixed constant (`0.0159`)
* This value was determined based on the ratio of head contributions from exploratory runs and is intentionally hardcoded within CAE.py
* Future researches may find, and chose to use, a more optimal value for their data though we found this value sufficed for model convergence. 

---

## Training Procedure

Training is performed using **cycle consistency** with a classifier-based conditioning signal.

For each minibatch:

### Context sampling

* **Source context**: real metadata for each cell
* **Target context**: for each metadata field, a new value is randomly sampled per cell

---

### First cycle (source → target)

* Encode expression using source contexts
* Decode using target contexts
* Cache hidden representations

#### Target-consistency loss

* A classifier predicts metadata from generated expression
* Cross-entropy against the *target* context provides the conditioning signal
* The source code often refers to this as adversarial/adv loss. This is a misnomer as the idea was inspired by GAN models. It is important to not this is **NOT a GAN** model.

---

### Second cycle (target → source)

* Generated expression is passed back using reversed contexts
* Hidden representations are cached again


---

### Loss terms

* Reconstruction loss
* Integration loss (hidden-state consistency across cycles)
* Target-consistency loss
* Supervised classifier loss on real expression

All generator losses are backpropagated together.

---

## Interpretability and Use Cases

The model is explicitly **additive and interpretable**:

* Each metadata field contributes its own logit term
* The L2 norm of each head’s logits quantifies that field’s influence
* Contributions can be compared across fields and epochs

The model also supports **counterfactual generation**, allowing expression profiles to be transformed across arbitrary combinations of metadata contexts.

