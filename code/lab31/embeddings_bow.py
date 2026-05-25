"""
embeddings_bow.py  –  Lab 3.1 Part 1 (Bag-of-Words)
=====================================================
Pipeline per story
------------------
1. Build a BoW vector for every word token in the story (binary / count,
   vocabulary fitted on training stories only to avoid leakage).
2. Lanczos-downsample from word-rate to TR-rate via get_downsampled().
3. Align X rows to Y length via align_X_to_Y_lengths().
4. Trim edges and apply delays [1,2,3,4] via apply_trim_and_lag().
5. Save to results/design_matrices/X_bow_{subject}_{split}.npy

Why downsampling?
-----------------
Words arrive at roughly one per 200-300 ms while fMRI TRs are ~2 s apart.
Lanczos interpolation low-pass filters the word-rate signal and resamples it
at TR-rate so the stimulus matrix X has the same time axis as Y.

Why make_delayed?
-----------------
The haemodynamic response function (HRF) peaks ~4-6 s after neural activity.
Stacking delays [1,2,3,4] TRs (2-8 s) lets the linear model learn the
optimal HRF shape without having to pre-specify it.
"""

import os
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# ── shared utilities ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from preprocess import get_downsampled, align_X_to_Y_lengths, apply_trim_and_lag
from split import TRAIN_STORIES, TEST_STORIES

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)
# DM_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'design_matrices')
DM_DIR = os.path.join(os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'results')), 'design_matrices')
os.makedirs(DM_DIR, exist_ok=True)

SUBJECTS = ['subject2', 'subject3']


def load_raw_text():
    """Load story tokens and timing information from the shared project data."""
    path = os.path.join(DATA_DIR, 'raw_text.pkl')
    return pickle.load(open(path, 'rb'))


def fit_vectorizer(train_stories, wordseqs):
    """Fit CountVectorizer on training stories only to avoid leakage."""
    all_train_words = []
    for s in train_stories:
        all_train_words.append(' '.join(wordseqs[s].data))
    vectorizer = CountVectorizer(binary=False, lowercase=True, max_features=5000)
    vectorizer.fit(all_train_words)
    return vectorizer


def build_bow_vectors(stories, wordseqs, vectorizer):
    """
    Build a per-word BoW matrix for each story.
    Each word token gets its own row so Lanczos can interpolate
    from word-rate down to TR-rate.

    Returns:
        dict of {story: (n_words, vocab_size) float32 array}
    """
    word_vectors = {}
    for story in stories:
        words = wordseqs[story].data
        X = vectorizer.transform(words).toarray().astype(np.float32)
        word_vectors[story] = X
    return word_vectors


def process_split(stories, wordseqs, vectorizer, subject):
    """
    Full pipeline for one subject/split:
      1. BoW vectors (word-rate)
      2. Lanczos downsample to TR-rate
      3. Align to Y lengths
      4. Trim + lag

    Processes story-by-story to keep memory usage low.
    Returns stacked (T_total, vocab * n_delays) float32 array.
    """
    chunks = []
    for story in stories:
        print(f"    Processing: {story}")

        # 1. BoW for this story only
        bow = build_bow_vectors([story], wordseqs, vectorizer)

        # 2. Lanczos downsample
        ds = get_downsampled([story], bow, wordseqs)

        # 3. Align X to Y length
        ds = align_X_to_Y_lengths(ds, subject, DATA_DIR)

        # 4. Trim + lag
        processed = apply_trim_and_lag(ds)

        chunks.append(processed[story].astype(np.float32))
        del bow, ds, processed

    return np.vstack(chunks).astype(np.float32)


def stack_to_file(stories, wordseqs, vectorizer, subject, out_path):
    """
    Process stories one at a time and write directly into a memory-mapped
    file to avoid holding everything in RAM at once.
    """
    # First pass: count rows
    print("    Counting rows...")
    n_rows, n_cols = 0, None
    for story in stories:
        bow       = build_bow_vectors([story], wordseqs, vectorizer)
        ds        = get_downsampled([story], bow, wordseqs)
        ds        = align_X_to_Y_lengths(ds, subject, DATA_DIR)
        processed = apply_trim_and_lag(ds)
        x         = processed[story]
        n_rows   += x.shape[0]
        if n_cols is None:
            n_cols = x.shape[1]
        del bow, ds, processed, x

    print(f"    Total shape: ({n_rows}, {n_cols})")

    # Second pass: write memmap
    out = np.lib.format.open_memmap(out_path, mode='w+',
                                    dtype=np.float32,
                                    shape=(n_rows, n_cols))
    row = 0
    for story in stories:
        print(f"    Writing: {story}")
        bow       = build_bow_vectors([story], wordseqs, vectorizer)
        ds        = get_downsampled([story], bow, wordseqs)
        ds        = align_X_to_Y_lengths(ds, subject, DATA_DIR)
        processed = apply_trim_and_lag(ds)
        x         = processed[story].astype(np.float32)
        out[row:row + x.shape[0], :] = x
        row      += x.shape[0]
        del bow, ds, processed, x

    del out
    print(f"    Saved -> {out_path}")


def process_subject(subject, wordseqs, all_stories):
    """Build and save BoW train/test matrices for one subject."""
    print(f"\n=== {subject} ===")

    train_stories = [s for s in TRAIN_STORIES if s in all_stories]
    test_stories  = [s for s in TEST_STORIES  if s in all_stories]
    print(f"  Train stories: {len(train_stories)}  |  Test stories: {len(test_stories)}")

    # Fit vocabulary on train stories only so held-out test words do not shape
    # the feature space used for model selection/evaluation.
    vectorizer = fit_vectorizer(train_stories, wordseqs)

    train_path = os.path.join(DM_DIR, f'X_bow_{subject}_train.npy')
    test_path  = os.path.join(DM_DIR, f'X_bow_{subject}_test.npy')

    print("  Processing train stories...")
    stack_to_file(train_stories, wordseqs, vectorizer, subject, train_path)

    print("  Processing test stories...")
    stack_to_file(test_stories, wordseqs, vectorizer, subject, test_path)


def main():
    """Entry point used by the Lab 3.1 batch script."""
    wordseqs    = load_raw_text()
    all_stories = set(wordseqs.keys())

    for subject in SUBJECTS:
        process_subject(subject, wordseqs, all_stories)

    print("\nDone.")


if __name__ == '__main__':
    main()
