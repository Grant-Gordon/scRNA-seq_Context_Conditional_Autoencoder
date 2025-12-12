# Context-Conditioned Cycle Autoencoder for scRNA-seq
[Jump to Usage section](#usage)

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
├── dataloader.py                   # Chunk-level and batch-level datasets
├── trainer.py                      # Training loop and optimizer logic
├── logging_helpers.py              # TensorBoard + logging utilities
├── models/
│   ├── CAE.py                      # Core model architecture
│   └── context_classifier.py       # Metadata classifier
├── scripts/
│   ├── main.py                     # Primary Python entrypoint. Users will copy and edit the template to create this file
│   ├── metadata_preprocessor.py
│   ├── slurm_submission.sh         # SLURM job script,HPC entrypoint. Users will copy and edit the template to create this file
│   ├── template_main.py            # Template for main.py with necessary vars labeled EDIT_ME
│   └── template_slurm_submission.sh    # Template for slurm_submission.sh with necessary vars labeled EDIT_ME
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


# Usage

To train the model, there are only a handful of constants that must be changed before running; all of which are found within `scripts/template_main.py` and `scripts/template_slurm_submission.sh` (assuming this is being run on an HPC).

 The user should use these templates to create untemplated versions that they will edit.  
e.g. bash: `cp template_main.py main.py`  
Follow the steps below for which lines to edit. 

---
### 1) Edit `src/srcipts/main.py`

* Edit the name of your data directory and the glob pattern your data files follow  
(lines ~16-22)
```
#############################
#Data
OUTPUT_DIR=args.output_dir
DATA_DIR="EDIT_ME" #Absolute path to dir containing data chunks
META_GLOB="EDIT_ME" #.pkl glob  for metadata e.g. human_metadata_*.pkl
EXPR_GLOB="EDIT_ME"  #.npz glob for expression data e.g. human_counts_*.npz
```
  
---

* Next edit where preprocessed metadata belongs, and which fields to process  
(lines ~23-43)
```
    #Preprocessed metadata 
    RERUN_PREPROCESSOR=True
    PREPROCESSOR_DIR="EDIT_ME" #Path to preprocessed metadata (will be created if does not exist and RERUN_PREPROCESSOR=True)
    META_FIELDS_VOCABS_FILE_NAME="metadata_vocab_.json" #Should be unique for each run if running experiments with different INCLUDE_FIELDS, or datasets
    FIELD_SPECS_FILE_NAME="metadata_field_specs.json"#Should be unique for each run if running experiments with different INCLUDE_FIELDS or datasets
    META_FIELDS_VOCABS_PATH=f"{PREPROCESSOR_DIR}/{META_FIELDS_VOCABS_FILE_NAME}"  # { field_name: { value: idx, ... }, ... }   
    FIELD_SPECS_PATH=f"{PREPROCESSOR_DIR}/{FIELD_SPECS_FILE_NAME}"      #[ FieldSpec(field=..., cardinality=..., using=..., non_null_fraction=...), ... ]
    INCLUDE_FIELDS=[
        "EDIT_ME"
        #Common default:
        # "cell_type",
        # "disease",
        # "development_stage",
        # "dev_stage",
        # "sex",
        # "self_reported_ethnicity",
        # "tissue_general",
        # "tissue",
        # "assay"
    ]
```
---
* Lastly edit the training parameters for this experiment  
(lines ~54-60)

```
#Training
LEARNING_RATE="EDIT_ME" #e.g. 0.0001
BATCH_SIZE="EDIT_ME" #e.g. 128
NUM_EPOCHS="EDIT_ME" #e.g. 10
BATCH_WORKERS="EDIT_ME" #e.g. 2 #how many threads are used to preload batches in dataloaders
BATCH_PREFETCH_FACTOR="EDIT_ME" #e.g. 3 #how many batches each THREAD will attempt to preload into RAM in dataloaders
#Model
LATENT_DIM= "EDIT_ME" #e.g. 128
```
---

### 2) Edit `src/srcipts/slurm_submission.sh` ###

If you are running on an HPC you will likely also need to submit a slurm job. In theory training can be done exclusivly with the python entrypoint in `main.py` however, as this project was designed for HPC use, a SLURM entry point template has been provided as well. 

*It is worth noting, different HPC's have differnt configurations and capabilities. The provided template may not work on a Users HPC.*

---
* Lastly a User must simply edit the SBATCH directives, and project/venv paths
---
```
#!/bin/bash
#SBATCH --job-name=EDIT_ME
#SBATCH --output=EDIT_ME/job-outputs/Job_%j-%x/%x.out
#SBATCH --error=EDIT_ME/job-outputs/Job_%j-%x/%x.err
#SBATCH --time=EDIT_ME
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=EDIT_ME
```

---

```
PROJECT_ROOT_DIR=EDIT_ME #Base git repository e.g. /absolute/path/scRNA-seq_Context_Conditional_autoencoder
```

---


```
source EDIT_ME #venv e.g. /absoluete/path/venv/bin/activate
```
---

### Training should now run with the cmd: `sbatch slurm_submission.sh`