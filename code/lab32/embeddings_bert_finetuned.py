"""
embeddings_bert_finetuned.py - Fine-tuned BERT design matrices
==============================================================

This script loads the checkpoint saved by finetune_bert.py and reruns only the
embedding extraction/preprocessing step. It is useful when the checkpoint
already exists and the design matrices need to be regenerated without another
fine-tuning job.

Outputs:
    results/design_matrices/X_bert_finetuned_subj{2,3}_train.npy
    results/design_matrices/X_bert_finetuned_subj{2,3}_test.npy
"""

import sys
import os
import numpy as np
import pickle
import torch
from transformers import BertForMaskedLM, BertTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab31.split import TRAIN_STORIES, TEST_STORIES
from lab31.preprocess import get_downsampled, apply_trim_and_lag, align_X_to_Y_lengths
from finetune_bert import get_word_embeddings

DATA_DIR  = '/ocean/projects/mth250011p/shared/215a/final_project/data'
_RESULTS  = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
OUT_DIR   = os.path.join(_RESULTS, 'design_matrices')
CKPT_PATH = os.path.join(_RESULTS, 'models', 'bert_finetuned_checkpoint')

SUBJECTS = ['subject2', 'subject3']
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isdir(CKPT_PATH):
        raise FileNotFoundError(
            f'Checkpoint not found at {CKPT_PATH}. '
            'Run finetune_bert.py first to train and save the model.'
        )

    print('Loading wordseqs...')
    wordseqs    = pickle.load(open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb'))
    all_stories = TRAIN_STORIES + TEST_STORIES

    print(f'Loading fine-tuned checkpoint from {CKPT_PATH} on {DEVICE}...')
    tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')
    model     = BertForMaskedLM.from_pretrained(CKPT_PATH).to(DEVICE)

    # Reuse the same contextual chunking/pooling helper as the main fine-tuned
    # training script so checkpoint-only extraction matches the full pipeline.
    print('\nExtracting fine-tuned BERT embeddings...')
    word_vectors = get_word_embeddings(wordseqs, all_stories, model, tokenizer, DEVICE)

    for subject in SUBJECTS:
        print(f'\n=== {subject} ===')

        # From this point on, preprocessing is identical to all other
        # embeddings: downsample to TRs, validate X/Y alignment, trim, and add
        # HRF-delay copies.
        print('Downsampling from word-rate to TR-rate via Lanczos filter...')
        X_down = get_downsampled(all_stories, word_vectors, wordseqs)

        print('Aligning X to Y lengths...')
        X_aligned = align_X_to_Y_lengths(X_down, subject, DATA_DIR)

        print('Trimming edges and applying HRF delays...')
        X_proc = apply_trim_and_lag(X_aligned)

        X_train = np.vstack([X_proc[s] for s in TRAIN_STORIES])
        X_test  = np.vstack([X_proc[s] for s in TEST_STORIES])

        subj_id    = subject.replace('subject', '')
        train_path = os.path.join(OUT_DIR, f'X_bert_finetuned_subj{subj_id}_train.npy')
        test_path  = os.path.join(OUT_DIR, f'X_bert_finetuned_subj{subj_id}_test.npy')

        np.save(train_path, X_train)
        np.save(test_path,  X_test)

        print(f'Saved: {train_path} {X_train.shape}')
        print(f'Saved: {test_path}  {X_test.shape}')

    print('\nDone.')
