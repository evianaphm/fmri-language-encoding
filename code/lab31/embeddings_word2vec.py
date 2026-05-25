"""
embeddings_word2vec.py - Lab 3.1 Word2Vec design matrices
==========================================================

This script builds the Word2Vec static-semantic baseline. Each story token is
mapped to a 300-dimensional Google News Word2Vec vector, then the shared Lab 3
preprocessing pipeline converts word-rate features into delayed TR-rate design
matrices for each subject.

Like GloVe, Word2Vec is context-independent. Comparing these matrices against
BERT later lets us separate "static semantic similarity" from contextual story
meaning.
"""

import sys
import os

# Add the repo root to path so we can import from code/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')) 
# Add the code/ directory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from split import TRAIN_STORIES, TEST_STORIES
import pickle
import numpy as np
import gensim.downloader as api
from preprocess import get_downsampled, apply_trim_and_lag, align_X_to_Y_lengths

DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)
OUT_DIR = os.path.join(os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'results')), 'design_matrices')
os.makedirs(OUT_DIR, exist_ok=True)

def story_to_vecs(words, model, dim=300):
    """
    Convert a list of story tokens to a Word2Vec feature matrix.

    Args:
        words: list of strings
        model: pretrained Word2Vec model (gensim KeyedVectors)
        dim:   dimensionality of the word vectors (default 300 for Google News)

    Returns:
        (n_words, dim) float32 array of word vectors, with zeros for OOV
    """
    # Start with zeros so out-of-vocabulary words remain represented but do
    # not introduce arbitrary values into the design matrix.
    vecs = np.zeros((len(words), dim), dtype=np.float32)
    for i, word in enumerate(words):
        if word in model:
            vecs[i] = model[word]
        # The Google News vocabulary is case sensitive, so try a lowercase
        # fallback before leaving the row as zeros.
        elif word.lower() in model:
            vecs[i] = model[word.lower()]
    return vecs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # raw_text.pkl supplies both story tokens and the timing information needed
    # to downsample word-level vectors to the fMRI TR grid.
    with open(os.path.join(DATA_DIR, "raw_text.pkl"), "rb") as f:
        wordseqs = pickle.load(f)

    # gensim downloads the model on first use and then loads it from cache.
    print("Loading pretrained Word2Vec model...")
    w2v = api.load("word2vec-google-news-300")

    all_stories = TRAIN_STORIES + TEST_STORIES

    # Build one word-rate matrix per story. Downsampling happens after this so
    # the timing from raw_text.pkl stays attached to the original word tokens.
    word_vectors = {}
    for story in all_stories:
        words = wordseqs[story].data
        word_vectors[story] = story_to_vecs(words, w2v, dim=300)
        print(story, word_vectors[story].shape)

    # Convert each story from word-level to TR-level, align to the subject's
    # BOLD response length, and append HRF-delay copies of the features.
    X_ds = get_downsampled(all_stories, word_vectors, wordseqs)
    for subject in ["subject2", "subject3"]:
        # Keep only stories with a response matrix for this subject.
        valid = set(
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(DATA_DIR, subject))
            if f.endswith('.npy')
        )
        train_stories = [s for s in TRAIN_STORIES if s in valid]
        test_stories  = [s for s in TEST_STORIES  if s in valid]

        X_aligned = align_X_to_Y_lengths(X_ds, subject, DATA_DIR)
        X_proc = apply_trim_and_lag(X_aligned)

        # Stack story matrices in the same order used by the ridge script when
        # it stacks the corresponding fMRI responses.
        X_train = np.vstack([X_proc[s] for s in train_stories])
        X_test = np.vstack([X_proc[s] for s in test_stories])

        print(subject)

        print("X_train shape:", X_train.shape) # (27111, 1200)
        print("X_test shape:", X_test.shape) # (6968, 1200)

        np.save(f"{OUT_DIR}/X_word2vec_{subject}_train.npy", X_train)
        np.save(f"{OUT_DIR}/X_word2vec_{subject}_test.npy", X_test)

    if not os.path.exists(f"{OUT_DIR}/stories_train.pkl"):
        with open(f"{OUT_DIR}/stories_train.pkl", "wb") as f:
            pickle.dump(TRAIN_STORIES, f)

    if not os.path.exists(f"{OUT_DIR}/stories_test.pkl"):
        with open(f"{OUT_DIR}/stories_test.pkl", "wb") as f:
            pickle.dump(TEST_STORIES, f)

    print(f"Saved outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()
