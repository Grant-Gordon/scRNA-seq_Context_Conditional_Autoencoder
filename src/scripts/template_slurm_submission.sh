#!/bin/bash
#SBATCH --job-name=EDIT_ME
#SBATCH --output=EDIT_ME/job-outputs/Job_%j-%x/%x.out
#SBATCH --error=EDIT_ME/job-outputs/Job_%j-%x/%x.err
#SBATCH --time=EDIT_ME
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=EDIT_ME

set -euo pipefail #shopts for error handling


PROJECT_ROOT_DIR=EDIT_ME #Base git repository e.g. /absolute/path/scRNA-seq_Context_Conditional_autoencoder
OUTPUT_DIR="${PROJECT_ROOT_DIR}/job-outputs/Job_${SLURM_JOB_ID}-${SLURM_JOB_NAME}"


cd "$GIT_ROOT_DIR" || exit 1
source EDIT_ME #venv e.g. /absoluete/path/venv/bin/activate
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PROJECT_ROOT_DIR/src:$PYTHONPATH"
cd $PROJECT_ROOT_DIR ||exit 1

python3 -m scripts.main --output-dir="$OUTPUT_DIR"
