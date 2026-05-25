#!/bin/bash
#SBATCH --job-name=lora_stab
#SBATCH --output=lora_stab_%j.out
#SBATCH --error=lora_stab_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
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

module load anaconda3
source activate stat214-lab3

export DATA_DIR=/ocean/projects/mth250011p/shared/215a/final_project/data
export RESULTS_DIR=/ocean/projects/mth250011p/$USER/results

mkdir -p $RESULTS_DIR/design_matrices
mkdir -p $RESULTS_DIR/models

echo "=== LoRA rank stability: fine-tune r=4 and r=16, extract design matrices ==="
python lab32/finetune_lora_stability.py

echo "=== Done! ==="
