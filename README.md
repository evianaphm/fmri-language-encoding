# Connecting Language to fMRI Responses

## Overview

This project builds encoding models that predict fMRI BOLD responses while people listen to natural spoken stories. The main goal is to compare how different text representations explain language-evoked brain activity, from simple lexical features to contextual BERT embeddings.

The pipeline has three main parts:

1. Create bag-of-words, GloVe, and Word2Vec design matrices, then fit voxel-wise ridge models.
2. Create pretrained BERT, fine-tuned BERT, and LoRA-adapted BERT design matrices, then compare model performance across embedding families.
3. Use wrapper-style SHAP and LIME analyses to interpret high-performing voxels by perturbing story text and rerunning the full prediction pipeline.

The project also includes stability analyses for two modeling choices: preserving surrounding story context when extracting BERT embeddings, and varying the LoRA adapter rank.

---

## Repository Structure

```text
├── code/                                      # All scripts and SLURM launchers
│   ├── run.sh                                # Submit the full pipeline with SLURM dependencies
│   ├── run_lab31.sh                          # Lexical/static embedding pipeline
│   ├── run_lab32_embeddings.sh               # BERT-family embedding/design-matrix generation
│   ├── run_lab32_ridge_plots.sh              # BERT-family ridge regression + plots
│   ├── run_stability_wordlevel_embeddings.sh # Word-level BERT stability embeddings
│   ├── run_stability_wordlevel_ridge_plots.sh # Word-level BERT stability ridge + plots
│   ├── run_stability_lora_rank_embeddings.sh # LoRA rank-stability embeddings
│   ├── run_stability_lora_rank_ridge.sh      # LoRA rank-stability ridge
│   ├── run_lab33_shap_lime.sh                # SHAP/LIME interpretation
│   │
│   ├── lab31/                                # BoW, GloVe, Word2Vec pipeline
│   ├── lab32/                                # BERT, fine-tuning, LoRA, ridge, stability plots
│   ├── lab33/                                # SHAP/LIME interpretation pipeline
│   ├── preprocessing.py                      # Shared downsampling and delay utilities
│   └── environment.yaml                      # Conda environment specification
│
├── ridge_utils/                              # Ridge regression and stimulus utilities
├── figures/                                  # Generated report figures
├── report/                                   # LaTeX report and compiled PDF
└── documents/                                # Reference paper(s)
```

Large intermediate files, model checkpoints, design matrices, ridge weights, and raw fMRI data are not stored in Git. They should be stored in an external results directory, for example:

```bash
/ocean/projects/<project>/<username>/results
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/evianaphm/fmri-language-encoding.git
cd fmri-language-encoding
```

### 2. Create or update the conda environment

```bash
conda env create -f code/environment.yaml
conda activate fmri-language-encoding
```

If the environment already exists:

```bash
conda env update -f code/environment.yaml
conda activate fmri-language-encoding
```

### 3. Configure data and output paths

The pipeline expects the following environment variables:

```bash
DATA_DIR=/path/to/fmri_story_data
RESULTS_DIR=/path/to/generated_results
```

The original analyses were run on a SLURM-managed HPC system. The run scripts set these variables internally for that environment, but they can be edited for a different local or cluster setup.

---

## Running the Pipeline

Most heavy jobs should be run on a compute cluster using SLURM. Run commands from the repository root.

### Option 1: submit the full pipeline

```bash
chmod +x code/run.sh
./code/run.sh
```

This submits separate SLURM jobs with dependencies so memory-heavy steps do not run in the same job. BERT-family embeddings are generated before ridge regression, and SHAP/LIME interpretation waits for the relevant ridge outputs.

### Option 2: run each stage separately

#### Lexical and Static Embedding Models

```bash
sbatch code/run_lab31.sh
```

Outputs:
- design matrices in `$RESULTS_DIR/design_matrices/`
- ridge outputs, metrics, and summary tables in `$RESULTS_DIR/ridge_31/`
- figures in `figures/lab31/`

#### BERT-Family Embeddings

```bash
sbatch code/run_lab32_embeddings.sh
```

This creates pretrained BERT, full fine-tuned BERT, and LoRA-adapted BERT design matrices. It does not run ridge regression.

Outputs:
- BERT design matrices in `$RESULTS_DIR/design_matrices/`
- fine-tuned model checkpoints in `$RESULTS_DIR/models/`

#### BERT-Family Ridge Regression and Plots

```bash
sbatch code/run_lab32_ridge_plots.sh
```

This fits voxel-wise ridge models for the BERT-family embeddings and regenerates the comparison figures.

Outputs:
- ridge outputs, metrics, and summary tables in `$RESULTS_DIR/ridge_32/`
- figures in `figures/lab32/`

#### Stability: Word-Level BERT Baseline

```bash
sbatch code/run_stability_wordlevel_embeddings.sh
sbatch code/run_stability_wordlevel_ridge_plots.sh
```

This compares the main contextual BERT embedding pipeline against a word-level BERT baseline with much less surrounding story context.

Outputs:
- word-level BERT design matrices in `$RESULTS_DIR/design_matrices/`
- ridge outputs in `$RESULTS_DIR/ridge_32_stability/`
- stability figures in `figures/lab32/`

#### Stability: LoRA Rank

```bash
sbatch code/run_stability_lora_rank_embeddings.sh
sbatch code/run_stability_lora_rank_ridge.sh
```

This trains and evaluates additional LoRA ranks for the rank-sensitivity stability check.

Outputs:
- LoRA rank-stability design matrices in `$RESULTS_DIR/design_matrices/`
- ridge outputs in `$RESULTS_DIR/ridge_32_lora_stability/`
- stability figures in `figures/lab32/`

#### SHAP and LIME Interpretation

```bash
sbatch code/run_lab33_shap_lime.sh
```

This runs wrapper-style SHAP and LIME analyses using the fine-tuned BERT ridge model for selected high-CC voxels and held-out stories.

Outputs:
- SHAP/LIME figures in `figures/lab33/`

---

## Generated Outputs

Generated outputs are intentionally excluded from Git because they can be very large. A complete run produces a results directory with the following structure:

```text
results/
│
├── design_matrices/              # Processed X matrices used for ridge regression
│   ├── X_bow_subj*_train/test.npz
│   ├── X_glove_subj*_train/test.npy
│   ├── X_word2vec_subj*_train/test.npy
│   ├── X_bert_pretrained_subj*_train/test.npy
│   ├── X_bert_finetuned_subj*_train/test.npy
│   ├── X_bert_lora_subj*_train/test.npy
│   ├── X_bert_pretrained_wordlevel_subj*_train/test.npy
│   └── X_bert_lora_r{4,16}_subj*_train/test.npy
│
├── models/                       # Fine-tuned BERT and LoRA checkpoints
│   ├── bert_finetuned_checkpoint/
│   ├── bert_lora_checkpoint/
│   └── bert_lora_r{4,16}_checkpoint/
│
├── ridge_31/                     # Ridge outputs for BoW, GloVe, Word2Vec
│   ├── *_weights.npz
│   ├── *_model.pkl
│   ├── *_corrs.npy
│   ├── *_valphas.npy
│   ├── *_stats.json
│   └── all_results.*
│
├── ridge_32/                     # Ridge outputs for BERT, BERT-FT, LoRA
│   ├── *_weights.npz
│   ├── *_model.pkl
│   ├── *_corrs.npy
│   ├── *_valphas.npy
│   ├── *_boot_corrs.npz
│   ├── *_stats.json
│   └── all_results.*
│
├── ridge_32_stability/           # Word-level BERT stability outputs
├── ridge_32_lora_stability/      # LoRA rank-stability outputs
└── tmp/                          # Temporary memmaps used during ridge fitting
```

The repository itself contains the code, report figures, and report materials needed to understand and reproduce the analysis.

---

## Methods Summary

All embedding methods are downsampled to the fMRI TR grid, aligned to the BOLD responses, trimmed, and expanded with HRF delays. Voxel-wise ridge regression is then fit separately for each subject and embedding method. Prediction performance is evaluated on held-out stories using the Pearson correlation coefficient between predicted and observed BOLD responses for each voxel.

For interpretation, `code/lab33/interpret_shap_lime.py` uses a wrapper around the full prediction pipeline. Word perturbations are passed through:

```text
perturbed story → BERT embeddings → TR downsampling → trimming/alignment → HRF delays → ridge prediction → story-level CC
```

This means SHAP/LIME word importance reflects changes in story-level prediction quality for a selected voxel, not direct neural activation to an isolated word.

---

## Troubleshooting

**Import errors**

Make sure the conda environment is active:

```bash
conda activate fmri-language-encoding
```

Also make sure you are running from the repository root when submitting SLURM jobs.

**Missing design matrices**

If ridge scripts fail with missing `X_*.npy` files, run the corresponding embedding script first:

```bash
sbatch code/run_lab32_embeddings.sh
```

or, for stability checks:

```bash
sbatch code/run_stability_wordlevel_embeddings.sh
sbatch code/run_stability_lora_rank_embeddings.sh
```

**Out-of-memory errors**

Embedding/fine-tuning jobs and ridge jobs are intentionally separated. If a ridge job runs out of memory, resubmit only the ridge script rather than rerunning embeddings:

```bash
sbatch code/run_lab32_ridge_plots.sh
```

**SHAP/LIME is slow**

SHAP/LIME reruns BERT and the preprocessing pipeline many times for each selected voxel. Use the SLURM script:

```bash
sbatch code/run_lab33_shap_lime.sh
```

Do not run SHAP/LIME on a login node.
