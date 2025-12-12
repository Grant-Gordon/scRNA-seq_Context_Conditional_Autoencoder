#!/bin/bash
#SBATCH --job-name=clean_test
#SBATCH --output=/mnt/home/gordongr/research/scRNA-seq_Context_Conditional_Autoencoder/job-outputs/Job_%j-%x/%x.out
#SBATCH --error=/mnt/home/gordongr/research/scRNA-seq_Context_Conditional_Autoencoder//job-outputs/Job_%j-%x/%x.err
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=32G

set -euo pipefail #shopts for error handling


GIT_ROOT_DIR="/mnt/home/gordongr/research/scRNA-seq_Context_Conditional_Autoencoder"
PROJECT_ROOT_DIR="${GIT_ROOT_DIR}"
OUTPUT_DIR="${PROJECT_ROOT_DIR}/job-outputs/Job_${SLURM_JOB_ID}-${SLURM_JOB_NAME}"


cd "$GIT_ROOT_DIR" || exit 1
source /mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/personalVAEenv/bin/activate
mkdir -p $OUTPUT_DIR
export PYTHONPATH="$PROJECT_ROOT_DIR/src:$PYTHONPATH"
cd $PROJECT_ROOT_DIR ||exit 1

python3 -m scripts.main --output-dir="$OUTPUT_DIR"
