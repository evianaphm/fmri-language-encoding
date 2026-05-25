#!/bin/bash
# run from repo root with:
#   chmod +x code/run.sh
#   ./code/run.sh

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Submitting full Lab 3 pipeline..."

LAB31_JOB=$(sbatch --parsable code/run_lab31.sh)
echo "Submitted Lab 3.1 full pipeline: $LAB31_JOB"

LAB32_EMB_JOB=$(sbatch --parsable code/run_lab32_embeddings.sh)
echo "Submitted Lab 3.2 embeddings: $LAB32_EMB_JOB"

LAB32_RIDGE_JOB=$(sbatch --parsable --dependency=afterok:$LAB32_EMB_JOB code/run_lab32_ridge_plots.sh)
echo "Submitted Lab 3.2 ridge + plots after embeddings: $LAB32_RIDGE_JOB"

WORD_EMB_JOB=$(sbatch --parsable code/run_stability_wordlevel_embeddings.sh)
echo "Submitted word-level BERT stability embeddings: $WORD_EMB_JOB"

WORD_RIDGE_JOB=$(sbatch --parsable --dependency=afterok:$WORD_EMB_JOB:$LAB32_RIDGE_JOB code/run_stability_wordlevel_ridge_plots.sh)
echo "Submitted word-level BERT stability ridge + plots: $WORD_RIDGE_JOB"

LORA_EMB_JOB=$(sbatch --parsable code/run_stability_lora_rank_embeddings.sh)
echo "Submitted LoRA rank stability embeddings: $LORA_EMB_JOB"

LORA_RIDGE_JOB=$(sbatch --parsable --dependency=afterok:$LORA_EMB_JOB:$LAB32_RIDGE_JOB code/run_stability_lora_rank_ridge.sh)
echo "Submitted LoRA rank stability ridge: $LORA_RIDGE_JOB"

SHAP_JOB=$(sbatch --parsable --dependency=afterok:$LAB32_RIDGE_JOB code/run_lab33_shap_lime.sh)
echo "Submitted Lab 3.3 SHAP/LIME after Lab 3.2 ridge: $SHAP_JOB"

echo
echo "All jobs submitted."
echo "Check status with:"
echo "  squeue -u \$USER"
