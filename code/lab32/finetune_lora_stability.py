# code/lab32/finetune_lora_stability.py
#
# LoRA rank stability analysis for Lab 3.2 / PCS.
#
# Trains additional LoRA models at ranks r=4 and r=16, extracts contextual
# BERT embeddings, and saves design matrices under separate embedding names:
#
#   X_bert_lora_r4_subj{2,3}_{train,test}.npy
#   X_bert_lora_r16_subj{2,3}_{train,test}.npy
#
# The main LoRA run uses r=8, alpha=32. Here alpha is set to 4*r so that
# alpha/r stays fixed at 4 and rank is the main judgment call being varied.

import os
import sys
import pickle
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizerFast
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab31.split import TRAIN_STORIES, TEST_STORIES
from lab31.preprocess import get_downsampled, apply_trim_and_lag, align_X_to_Y_lengths
from lab32.finetune_bert import build_dataloader, train_bert
from lab32.finetune_lora import get_word_embeddings


DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)
RESULTS_DIR = os.environ.get(
    'RESULTS_DIR',
    os.path.join(os.path.dirname(__file__), '..', '..', 'results'),
)
OUT_DIR = os.path.join(RESULTS_DIR, 'design_matrices')
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')

SUBJECTS = ['subject2', 'subject3']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

RANKS = [
    int(rank.strip())
    for rank in os.environ.get('LORA_RANKS', '4,16').split(',')
    if rank.strip()
]
# Ranks can be overridden from the batch script with LORA_RANKS, but the
# default compares lower/higher capacity adapters against the main r=8 run.
LORA_DROPOUT = 0.1
LORA_MODULES = ['query', 'value']
EPOCHS = 3
LR = 5e-4


def set_seed(seed=SEED):
    """Set NumPy/PyTorch seeds; each rank gets a deterministic rank-specific seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lora_model(rank: int):
    """
    Build a LoRA-wrapped BERT model for one rank-stability run.

    The scaling alpha is set to 4 * rank so alpha/r remains fixed. This keeps
    the comparison focused on adapter rank rather than changing both capacity
    and update scaling at the same time.
    """
    alpha = 4 * rank
    base_model = BertForMaskedLM.from_pretrained('google-bert/bert-base-uncased')
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=LORA_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias='none',
    )
    model = get_peft_model(base_model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f'LoRA r={rank}, alpha={alpha}: '
        f'{trainable:,} / {total:,} trainable params '
        f'({100 * trainable / total:.4f}%)'
    )
    return model


def save_design_matrices(emb_name: str, word_vectors: dict, wordseqs: dict, all_stories: list):
    """
    Convert rank-specific word embeddings into subject-specific design matrices.

    The word embeddings are shared across subjects because the stories are the
    same; the alignment step is subject-specific because fMRI story lengths can
    differ after trimming.
    """
    for subject in SUBJECTS:
        print(f'\n=== {subject} / {emb_name} ===')

        print('Downsampling from word-rate to TR-rate via Lanczos filter...')
        X_down = get_downsampled(all_stories, word_vectors, wordseqs)

        print('Aligning X to Y lengths...')
        X_aligned = align_X_to_Y_lengths(X_down, subject, DATA_DIR)

        print('Trimming edges and applying HRF delays...')
        X_proc = apply_trim_and_lag(X_aligned)

        X_train = np.vstack([X_proc[s] for s in TRAIN_STORIES])
        X_test = np.vstack([X_proc[s] for s in TEST_STORIES])

        subj_id = subject.replace('subject', '')
        train_path = os.path.join(OUT_DIR, f'X_{emb_name}_subj{subj_id}_train.npy')
        test_path = os.path.join(OUT_DIR, f'X_{emb_name}_subj{subj_id}_test.npy')

        np.save(train_path, X_train)
        np.save(test_path, X_test)

        print(f'Saved: {train_path} {X_train.shape}')
        print(f'Saved: {test_path}  {X_test.shape}')


def run_rank(rank: int, wordseqs: dict, all_stories: list, tokenizer):
    """Train one LoRA rank, extract embeddings, and save its ridge-ready matrices."""
    emb_name = f'bert_lora_r{rank}'
    print('\n' + '=' * 72)
    print(f'Running LoRA stability rank r={rank} -> {emb_name}')
    print('=' * 72)

    set_seed(SEED + rank)
    model = build_lora_model(rank).to(DEVICE)

    print('\nBuilding dataloader from training stories only (no leakage)...')
    dataloader = build_dataloader(wordseqs, TRAIN_STORIES, tokenizer)

    print(f'\nFine-tuning LoRA r={rank} via MLM...')
    train_bert(model, dataloader, tokenizer, epochs=EPOCHS, lr=LR, device=DEVICE)

    ckpt_path = os.path.join(MODEL_DIR, f'{emb_name}_checkpoint')
    model.save_pretrained(ckpt_path)
    print(f'\nSaved LoRA r={rank} checkpoint -> {ckpt_path}')

    print(f'\nExtracting contextual embeddings for {emb_name}...')
    word_vectors = get_word_embeddings(wordseqs, all_stories, model, tokenizer, DEVICE)
    save_design_matrices(emb_name, word_vectors, wordseqs, all_stories)

    del model, word_vectors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('Loading wordseqs...')
    with open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb') as fh:
        wordseqs = pickle.load(fh)
    all_stories = TRAIN_STORIES + TEST_STORIES

    print('Loading tokenizer...')
    tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')

    for rank in RANKS:
        run_rank(rank, wordseqs, all_stories, tokenizer)

    print('\nDone.')


if __name__ == '__main__':
    main()
