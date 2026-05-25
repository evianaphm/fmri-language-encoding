"""
preprocess.py - Shared Lab 3 stimulus/BOLD alignment helpers
============================================================

All embedding methods use the same preprocessing contract:

    word-level features -> Lanczos downsampling -> X[5:-10] alignment
    -> delayed copies at [1, 2, 3, 4] TRs.

The fMRI response matrix Y is intentionally left untrimmed. The downsampled
stimulus matrix has 15 extra edge rows, so trimming X by 5 TRs at the start and
10 TRs at the end is what makes X and raw Y line up.
"""

import sys
import os
import numpy as np

# Add the repo root to path so we can import from code/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the code/ directory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing import downsample_word_vectors, make_delayed

# Constants verified from data exploration on Bridges2.
# Downsampled X has 15 extra TR rows relative to raw Y, so X[5:-10]
# is the alignment step that makes the stimulus matrix match Y.
TR = 2.0          # seconds per fMRI volume
TRIM_START = 5    # TRs
TRIM_END = 10     # TRs
DELAYS = [1, 2, 3, 4]  # in TRs


def get_downsampled(stories, word_vectors, wordseqs):
    """
    Downsample word-level embeddings to fMRI TR rate using Lanczos interpolation.

    Args:
        stories:      list of story names
        word_vectors: dict of {story: (n_words, d) float32 array}
        wordseqs:     dict of {story: DataSequence} — loaded from raw_text.pkl

    Returns:
        dict of {story: (n_trs, d) array}
    """
    return downsample_word_vectors(stories, word_vectors, wordseqs)


def apply_trim_and_lag(X_dict):
    """
    Trim edge TRs and concatenate lagged copies of features.

    Trimming removes edge artifacts where the BOLD signal does not
    cleanly correspond to the stimulus. Lagging accounts for the
    hemodynamic response function (HRF) — the brain's BOLD signal
    peaks ~4-6 seconds after a stimulus, so we concatenate shifted
    copies of X at delays [1,2,3,4] TRs and let the linear model
    learn the right delay automatically.

    Args:
        X_dict: dict of {story: (n_trs, d) array}

    Returns:
        dict of {story: (n_trs - TRIM_START - TRIM_END, d * 4) array}
    """
    processed = {}
    for story, X in X_dict.items():
        X = X[TRIM_START : -TRIM_END]
        X = make_delayed(X, DELAYS)
        processed[story] = X
    return processed


def trim_Y(Y):
    """
    Return the raw Y (fMRI response) matrix.

    The 5/10 TR edge trim belongs to the downsampled stimulus matrix X:
    X_down has 15 extra rows relative to Y, and X_down[5:-10] aligns it
    to the raw fMRI response. Trimming Y here would over-trim both the
    train/test matrices and the interpretation stories.

    Args:
        Y: (n_trs, n_voxels) array

    Returns:
        (n_trs, n_voxels) array
    """
    return Y


def align_X_to_Y_lengths(X_dict, subject, data_dir):
    """
    Validate that trimming downsampled X will match raw Y row counts.

    Historically this function truncated X before apply_trim_and_lag().
    That caused an extra 15-TR trim because apply_trim_and_lag() then
    removed [5:-10] as well. The correct pipeline is:

        X_down -> X_down[5:-10] -> make_delayed
        Y_raw  -> unchanged

    Args: 
        X_dict: dict of {story: (n_trs, d) array}
        subject: string, e.g. "subject2"
        data_dir: path to data directory containing subject subdirs with Y .npy files

    Returns:
        The original X_dict, unchanged.
    """
    for story, X in X_dict.items():
        Y = np.load(os.path.join(data_dir, subject, f"{story}.npy"), mmap_mode="r")
        n_trs_y = Y.shape[0]
        n_trs_x_after_trim = X[TRIM_START:-TRIM_END].shape[0]
        if n_trs_x_after_trim != n_trs_y:
            raise ValueError(
                f"X/Y length mismatch for {subject}/{story}: "
                f"X_down[5:-10]={n_trs_x_after_trim}, Y_raw={n_trs_y}"
            )
    return X_dict

if __name__ == "__main__":
    # Quick shape verification — run this on Bridges2 to confirm
    # X and Y time axes align before handing off to Members B/C/D
    import numpy as np
    import pickle
    from split import TRAIN_STORIES, TEST_STORIES

    DATA_DIR = os.environ.get(
        'DATA_DIR',
        '/ocean/projects/mth250011p/shared/215a/final_project/data',
    )
    wordseqs = pickle.load(open(f'{DATA_DIR}/raw_text.pkl', 'rb'))

    print(f"Train stories ({len(TRAIN_STORIES)}): {TRAIN_STORIES}")
    print(f"Test stories ({len(TEST_STORIES)}):  {TEST_STORIES}")

    all_stories = TRAIN_STORIES + TEST_STORIES
    for subject in ['subject2', 'subject3']:
        print(f"\n--- {subject} ---")
        for story in all_stories:
            Y = np.load(f'{DATA_DIR}/{subject}/{story}.npy')
            Y_trim = trim_Y(Y)
            n_trs_trim = Y_trim.shape[0]
            n_trs_ws = len(wordseqs[story].tr_times)
            print(f"{story}: Y raw {Y.shape} → trimmed {Y_trim.shape} | wordseq trs: {n_trs_ws}")
            assert n_trs_trim <= n_trs_ws, f"MISMATCH on {story}"

    print("\nAll shapes verified.")
