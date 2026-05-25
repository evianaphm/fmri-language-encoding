#!/bin/bash
#SBATCH --job-name=bert_word_embed
#SBATCH --output=bert_word_embed_%j.out
#SBATCH --error=bert_word_embed_%j.err
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000M
#SBATCH --partition=RM-shared
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

mkdir -p $RESULTS_DIR/design_matrices
echo "=== Lab 3.2 stability: noncontextual word-level BERT embeddings only ==="
python lab32/embeddings_bert_pretrained_wordlevel.py

echo "=== Done! ==="
