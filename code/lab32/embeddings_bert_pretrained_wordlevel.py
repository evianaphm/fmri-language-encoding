# code/lab32/embeddings_bert_pretrained_wordlevel.py
"""
Extract pretrained BERT embeddings one word at a time.

This script is intended only for PCS/stability comparisons. Unlike the main
pretrained BERT pipeline, each word is encoded independently, so BERT does not
see the surrounding story context. The output is saved under a separate
embedding name and does not overwrite the main contextual BERT matrices:

    X_bert_pretrained_wordlevel_subj2_train.npy
    X_bert_pretrained_wordlevel_subj2_test.npy
    X_bert_pretrained_wordlevel_subj3_train.npy
    X_bert_pretrained_wordlevel_subj3_test.npy
"""

import os
import sys
import pickle
import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab31.split import TRAIN_STORIES, TEST_STORIES
from lab31.preprocess import get_downsampled, apply_trim_and_lag, align_X_to_Y_lengths


EMB_NAME = 'bert_pretrained_wordlevel'
DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)
RESULTS_DIR = os.environ.get(
    'RESULTS_DIR',
    os.path.join(os.path.dirname(__file__), '..', '..', 'results'),
)
OUT_DIR = os.path.join(RESULTS_DIR, 'design_matrices')
SUBJECTS = ['subject2', 'subject3']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _last_hidden_state(outputs):
    """Return the final hidden layer for both BertModel and BertForMaskedLM outputs."""
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    return outputs.hidden_states[-1]


def get_wordlevel_embeddings(wordseqs, stories, model, tokenizer, device):
    """
    Extract one BERT-derived vector per word with minimal surrounding context.

    This is deliberately different from the main contextual BERT pipeline:
    every word is tokenized and encoded independently. The comparison tests
    whether preserving story context during embedding extraction matters for
    downstream fMRI prediction.
    """
    model.eval()
    word_vectors = {}

    for story in stories:
        vectors = []

        with torch.no_grad():
            for word in wordseqs[story].data:
                if not word:
                    vectors.append(np.zeros(768, dtype=np.float32))
                    continue

                inputs = tokenizer(
                    word,
                    return_tensors='pt',
                    truncation=True,
                    max_length=16,
                ).to(device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden = _last_hidden_state(outputs)[0]

                # Average non-special token states for this isolated word.
                # If tokenization leaves only special tokens, fall back to the
                # full hidden-state average so the row is still well-defined.
                if hidden.shape[0] > 2:
                    emb = hidden[1:-1].mean(dim=0)
                else:
                    emb = hidden.mean(dim=0)
                vectors.append(emb.cpu().numpy().astype(np.float32))

        word_vectors[story] = np.vstack(vectors).astype(np.float32)
        print(f"  {story}: {word_vectors[story].shape}")

    return word_vectors


def main():
    print("Loading wordseqs...")
    with open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb') as fh:
        wordseqs = pickle.load(fh)

    print(f"Loading pretrained BERT on {DEVICE}...")
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased").to(DEVICE)

    all_stories = TRAIN_STORIES + TEST_STORIES
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Extracting low-context word-level pretrained BERT embeddings...")
    word_vectors = get_wordlevel_embeddings(
        wordseqs,
        all_stories,
        model,
        tokenizer,
        DEVICE,
    )

    for subject in SUBJECTS:
        print(f"\n=== {subject} ===")

        # Use the same downstream preprocessing as the main BERT pipeline so
        # the stability comparison isolates the context-extraction choice.
        print("Downsampling...")
        X_down = get_downsampled(all_stories, word_vectors, wordseqs)

        print("Aligning X to Y lengths...")
        X_down = align_X_to_Y_lengths(X_down, subject, DATA_DIR)

        print("Trimming and lagging...")
        X_proc = apply_trim_and_lag(X_down)

        X_train = np.vstack([X_proc[s] for s in TRAIN_STORIES])
        X_test = np.vstack([X_proc[s] for s in TEST_STORIES])

        subj_id = subject.replace('subject', '')
        train_path = os.path.join(OUT_DIR, f"X_{EMB_NAME}_subj{subj_id}_train.npy")
        test_path = os.path.join(OUT_DIR, f"X_{EMB_NAME}_subj{subj_id}_test.npy")

        np.save(train_path, X_train)
        np.save(test_path, X_test)

        print(f"Saved: {train_path} {X_train.shape}")
        print(f"Saved: {test_path}  {X_test.shape}")

    print("\nDone.")


if __name__ == '__main__':
    main()
