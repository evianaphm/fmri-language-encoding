#!/bin/bash
#SBATCH --job-name=lab31
#SBATCH --output=lab31_%j.out
#SBATCH --error=lab31_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --partition=RM
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
conda activate stat214-lab3
export DATA_DIR=/ocean/projects/mth250011p/shared/215a/final_project/data
export RESULTS_DIR=/ocean/projects/mth250011p/$USER/results

# Create output directories
mkdir -p $RESULTS_DIR/design_matrices
mkdir -p $RESULTS_DIR/ridge_31

echo "=== Cleaning old design matrices ==="
rm -f $RESULTS_DIR/design_matrices/X_bow_*.npy
rm -f $RESULTS_DIR/design_matrices/X_bow_*.npz
rm -f $RESULTS_DIR/design_matrices/X_glove_*.npy
rm -f $RESULTS_DIR/design_matrices/X_word2vec_*.npy

echo "=== Step 1: BoW embeddings ==="
python lab31/embeddings_bow.py

echo "=== Step 2: GloVe embeddings ==="
python lab31/embeddings_glove.py

echo "=== Step 3: Word2Vec embeddings ==="
python lab31/embeddings_word2vec.py

echo "=== Step 4: Ridge regression ==="
python lab31/ridge_31.py

echo "=== Step 5: Plots ==="
python lab31/plot_31.py

echo "=== Done! ==="
