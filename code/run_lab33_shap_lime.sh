#!/bin/bash
#SBATCH --job-name=shap_lime33
#SBATCH --output=shap_lime33_%j.out
#SBATCH --error=shap_lime33_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=RM-shared
#SBATCH --account=mth250011p

set -e
set -o pipefail

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
export PYTHONPATH=$REPO_ROOT/code:$REPO_ROOT:$PYTHONPATH

module load anaconda3
source activate stat214-lab3

export DATA_DIR=/ocean/projects/mth250011p/shared/215a/final_project/data
export RESULTS_DIR=/ocean/projects/mth250011p/$USER/results
export DEVICE=cpu

echo "=== Subject 2 + Subject 3: fine-tuned BERT SHAP/LIME ==="
SUBJECTS=subject2,subject3 EMB=bert_finetuned python code/lab33/interpret_shap_lime.py

echo "=== Done ==="
