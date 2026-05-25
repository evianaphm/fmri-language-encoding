"""
embeddings_glove.py - Lab 3.1 GloVe design matrices
====================================================

This script converts each story into a sequence of 300-dimensional GloVe
vectors, downsamples those word-level vectors to the fMRI TR grid, applies the
shared 5/10 TR alignment and HRF-delay preprocessing, and saves one train/test
design matrix per subject.

GloVe is context-independent: the same word receives the same vector regardless
of where it appears in the story. That makes this script the static-semantic
baseline used later to compare against contextual BERT embeddings.
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
    """Convert a list of words to a matrix of word vectors.

    Args:
        words: list of strings
        model: pretrained gensim KeyedVectors model
        dim: dimensionality of the word vectors

    Returns:
        (n_words, dim) float32 array of word vectors, with zeros for OOV words
    """
    vecs = np.zeros((len(words), dim), dtype=np.float32)
    for i, word in enumerate(words):
        if word in model:
            vecs[i] = model[word]
        elif word.lower() in model:
            vecs[i] = model[word.lower()]
    return vecs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # raw_text.pkl stores a DataSequence per story, including word tokens,
    # word onset times, and TR times needed for Lanczos downsampling.
    with open(os.path.join(DATA_DIR, "raw_text.pkl"), "rb") as f:
        wordseqs = pickle.load(f)

    # gensim downloads the model on first use and reuses the local cache after.
    print("Loading pretrained GloVe model...")
    glove = api.load("glove-wiki-gigaword-300")

    all_stories = TRAIN_STORIES + TEST_STORIES

    # Build a word-rate feature matrix before downsampling. Each row
    # corresponds to one story token, matching wordseqs[story].data_times.
    word_vectors = {}
    for story in all_stories:
        words = wordseqs[story].data
        word_vectors[story] = story_to_vecs(words, glove, dim=300)
        print(story, word_vectors[story].shape)

    # The fMRI model is fit at TR resolution, so every embedding sequence goes
    # through the same downsample -> align -> trim -> HRF-delay pipeline.
    X_ds = get_downsampled(all_stories, word_vectors, wordseqs)
    for subject in ["subject2", "subject3"]:
        # Some subjects may not have every possible story file. Filtering here
        # keeps the fixed split while avoiding missing-Y file errors.
        valid = set(
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(DATA_DIR, subject))
            if f.endswith('.npy')
        )
        train_stories = [s for s in TRAIN_STORIES if s in valid]
        test_stories  = [s for s in TEST_STORIES  if s in valid]

        X_aligned = align_X_to_Y_lengths(X_ds, subject, DATA_DIR)
        X_proc = apply_trim_and_lag(X_aligned)

        # Stack story matrices into one continuous train/test design matrix.
        # The same story order is used later when Y matrices are stacked.
        X_train = np.vstack([X_proc[s] for s in train_stories])
        X_test = np.vstack([X_proc[s] for s in test_stories])

        print(subject)
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)

        np.save(f"{OUT_DIR}/X_glove_{subject}_train.npy", X_train)
        np.save(f"{OUT_DIR}/X_glove_{subject}_test.npy", X_test)

    with open(f"{OUT_DIR}/stories_train.pkl", "wb") as f:
        pickle.dump(TRAIN_STORIES, f)

    with open(f"{OUT_DIR}/stories_test.pkl", "wb") as f:
        pickle.dump(TEST_STORIES, f)

    print(f"Saved outputs to {OUT_DIR}")

if __name__ == "__main__":
    main()
