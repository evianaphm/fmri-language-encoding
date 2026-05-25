#!/bin/bash
#SBATCH --job-name=lab32_bert
#SBATCH --output=lab32_%j.out
#SBATCH --error=lab32_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-16:1
#SBATCH --account=mth250011p

set -e
set -o pipefail

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SCRIPT_DIR="$REPO_ROOT/code"
cd "$SCRIPT_DIR"
export PYTHONPATH=$REPO_ROOT/code:$REPO_ROOT:$PYTHONPATH

# Load conda environment
module load anaconda3
source activate stat214-lab3

export DATA_DIR=/ocean/projects/mth250011p/shared/215a/final_project/data
export RESULTS_DIR=/ocean/projects/mth250011p/$USER/results

mkdir -p $RESULTS_DIR/design_matrices
mkdir -p $RESULTS_DIR/models

echo "=== Step 1: Pretrained BERT embeddings ==="
python lab32/embeddings_bert_pretrained.py

echo "=== Step 2: Full fine-tuned BERT embeddings ==="
python lab32/finetune_bert.py

echo "=== Step 3: LoRA fine-tuned BERT embeddings ==="
python lab32/finetune_lora.py

echo "=== Done! Lab 3.2 embeddings/design matrices only. Run code/run_lab32_ridge_plots.sh next for ridge + plots. ==="
