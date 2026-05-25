"""
embeddings_bert_pretrained.py - Contextual pretrained BERT design matrices
==========================================================================

This is the main pretrained BERT embedding pipeline. Each story is processed in
overlapping chunks so BERT can represent each word using surrounding narrative
context. The resulting word-level contextual vectors are then downsampled,
aligned, trimmed, lagged, and saved as subject-specific ridge design matrices.
"""

import sys
import os
import numpy as np
import pickle
import torch
from transformers import BertTokenizerFast, BertModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab31.split import TRAIN_STORIES, TEST_STORIES
from lab31.preprocess import get_downsampled, apply_trim_and_lag, align_X_to_Y_lengths

DATA_DIR = '/ocean/projects/mth250011p/shared/215a/final_project/data'
_RESULTS = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
OUT_DIR = os.path.join(_RESULTS, 'design_matrices')
SUBJECTS = ['subject2', 'subject3']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _last_hidden_state(outputs):
    """Return the final hidden layer for both BertModel and BertForMaskedLM outputs."""
    if hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    return outputs.hidden_states[-1]


def get_word_embeddings(wordseqs, stories, model, tokenizer, device, chunk_size=96, stride=48):
    """
    For each story, extract one 768-d BERT embedding per word token.

    Strategy: run BERT on overlapping story chunks so each token is represented
    in context, then average all subword states that belong to each original
    word. Overlap reduces hard boundary effects between chunks.

    Returns:
        dict of {story: (n_words, 768) float32 array}
    """
    model.eval()
    word_vectors = {}

    for story in stories:
        words = wordseqs[story].data
        embedding_sums = np.zeros((len(words), 768), dtype=np.float32)
        embedding_counts = np.zeros(len(words), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(words), stride):
                chunk_words = words[start:start + chunk_size]
                valid_positions = [i for i, word in enumerate(chunk_words) if word]
                if not valid_positions:
                    continue

                valid_words = [chunk_words[i] for i in valid_positions]
                inputs = tokenizer(
                    valid_words,
                    is_split_into_words=True,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                )
                # word_ids maps WordPiece tokens back to the original word
                # positions within this chunk, which lets us pool subword
                # pieces into one vector per story word.
                word_ids = inputs.word_ids(batch_index=0)
                inputs = inputs.to(device)

                outputs = model(**inputs, output_hidden_states=True)
                hidden = _last_hidden_state(outputs)[0]

                for local_idx, original_idx in enumerate(valid_positions):
                    # A word can split into multiple WordPieces. Averaging
                    # those token states gives a single feature vector that
                    # still reflects the full contextual BERT pass.
                    token_positions = [
                        tok_idx for tok_idx, word_id in enumerate(word_ids)
                        if word_id == local_idx
                    ]
                    if token_positions:
                        pooled = hidden[token_positions].mean(dim=0)
                        word_idx = start + original_idx
                        embedding_sums[word_idx] += pooled.cpu().numpy()
                        embedding_counts[word_idx] += 1.0

        # Because chunks overlap, many words are embedded more than once.
        # Averaging across all observed contexts softens chunk-boundary effects.
        nonzero = embedding_counts > 0
        embeddings = np.zeros_like(embedding_sums)
        embeddings[nonzero] = embedding_sums[nonzero] / embedding_counts[nonzero, None]
        word_vectors[story] = embeddings
        print(f"  {story}: {word_vectors[story].shape}")

    return word_vectors


if __name__ == "__main__":
    # raw_text.pkl provides word tokens and timing information for all stories.
    print("Loading wordseqs...")
    wordseqs = pickle.load(open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb'))

    # Load BERT
    print(f"Loading BERT on {DEVICE}...")
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    model = BertModel.from_pretrained("google-bert/bert-base-uncased").to(DEVICE)

    all_stories = TRAIN_STORIES + TEST_STORIES
    os.makedirs(OUT_DIR, exist_ok=True)

    # Word vectors are shared across subjects; only the final X/Y alignment is
    # subject-specific because each subject has a separate BOLD response matrix.
    print("Extracting BERT embeddings...")
    word_vectors = get_word_embeddings(
        wordseqs, all_stories, model, tokenizer, DEVICE
    )

    for subject in SUBJECTS:
        print(f"\n=== {subject} ===")

        # Downsample, align, trim, and lag via the same preprocessing pipeline
        # used by BoW/GloVe/Word2Vec, so later comparisons differ only in the
        # embedding representation.
        print("Downsampling...")
        X_down = get_downsampled(all_stories, word_vectors, wordseqs)

        print("Aligning X to Y lengths...")
        X_down = align_X_to_Y_lengths(X_down, subject, DATA_DIR)

        print("Trimming and lagging...")
        X_proc = apply_trim_and_lag(X_down)

        # Stack story-level matrices into the fixed train/test split used by
        # every encoding model.
        X_train = np.vstack([X_proc[s] for s in TRAIN_STORIES])
        X_test  = np.vstack([X_proc[s] for s in TEST_STORIES])

        subj_id = subject.replace('subject', '')
        train_path = os.path.join(OUT_DIR, f"X_bert_pretrained_subj{subj_id}_train.npy")
        test_path  = os.path.join(OUT_DIR, f"X_bert_pretrained_subj{subj_id}_test.npy")

        np.save(train_path, X_train)
        np.save(test_path,  X_test)

        print(f"Saved: {train_path} {X_train.shape}")
        print(f"Saved: {test_path}  {X_test.shape}")

    print("\nDone.")
