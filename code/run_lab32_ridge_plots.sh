#!/bin/bash
#SBATCH --job-name=lab32_ridge
#SBATCH --output=lab32_ridge_%j.out
#SBATCH --error=lab32_ridge_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=RM
#SBATCH --account=mth250011p

set -e
set -o pipefail

# When run via sbatch, BASH_SOURCE[0] resolves to a spool copy — use SLURM_SUBMIT_DIR instead
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

mkdir -p $RESULTS_DIR/ridge_32
mkdir -p $RESULTS_DIR/tmp/ridge32
mkdir -p $REPO_ROOT/results/metrics
mkdir -p $REPO_ROOT/figures/lab32

echo "=== Lab 3.2 CPU ridge regression ==="
python lab32/ridge_32.py

echo "=== Lab 3.2 plots ==="
python lab32/plot_32.py

echo "=== Done! ==="
