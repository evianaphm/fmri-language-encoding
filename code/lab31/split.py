"""
split.py - Fixed story split used across all Lab 3 models
=========================================================

The same train/test story split is reused for BoW, GloVe, Word2Vec, BERT,
BERT-FT, LoRA, and the stability checks. Keeping the split fixed ensures that
performance differences are attributable to embedding/model choices rather
than different held-out stories.
"""

import os
import pickle
import random

DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)

def get_valid_stories():
    """
    Return stories that have both stimulus text and fMRI responses.

    A story is valid only if it appears in raw_text.pkl and has a corresponding
    response matrix for every subject included in the analysis. This avoids
    silently comparing models on different story sets.
    """
    raw = pickle.load(open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb'))
    valid = set(raw.keys())
    for subj in ['subject2', 'subject3']:
        subj_dir = os.path.join(DATA_DIR, subj)
        if os.path.isdir(subj_dir):
            y_stories = set(f.replace('.npy', '') for f in os.listdir(subj_dir)
                            if f.endswith('.npy'))
            valid &= y_stories
    return sorted(valid)

def get_split(test_size=0.2, random_state=42):
    """
    Deterministically split valid stories into train and held-out test sets.

    The random seed is fixed so all downstream embedding methods use exactly
    the same held-out stories.
    """
    stories = get_valid_stories()
    rng = random.Random(random_state)
    stories_shuffled = stories.copy()
    rng.shuffle(stories_shuffled)
    n_test = max(1, int(len(stories_shuffled) * test_size))
    return stories_shuffled[n_test:], stories_shuffled[:n_test]

TRAIN_STORIES, TEST_STORIES = get_split()
