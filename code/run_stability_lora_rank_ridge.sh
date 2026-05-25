#!/bin/bash
#SBATCH --job-name=lora_stab_ridge
#SBATCH --output=lora_stab_ridge_%j.out
#SBATCH --error=lora_stab_ridge_%j.err
#SBATCH --time=7:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
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

module load anaconda3
source activate stat214-lab3

export DATA_DIR=/ocean/projects/mth250011p/shared/215a/final_project/data
export RESULTS_DIR=/ocean/projects/mth250011p/$USER/results

mkdir -p $RESULTS_DIR/ridge_32_lora_stability
mkdir -p $RESULTS_DIR/tmp/ridge32

echo "=== LoRA rank stability: CPU ridge regression ==="
python lab32/ridge_32_lora_stability.py

echo "=== Done! ==="
