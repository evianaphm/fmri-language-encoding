"""
ridge_31.py  –  Lab 3.1 Part 2: Ridge Regression
=================================================
Run from repo root:
    python code/lab31/ridge_31.py

Memory strategy (in order of impact)
--------------------------------------
1.  float32 everywhere  →  halves RAM vs float64
2.  Y is NEVER fully loaded:
      - built story-by-story into a temp float32 memmap on disk
      - z-scored IN-PLACE in voxel chunks (no second full copy)
      - passed directly to bootstrap_ridge (memmap is numpy-indexable)
3.  X (BoW) is sparse on disk (.npz) and in memory (CSR).
    It is densified only once, immediately before the SVD inside
    bootstrap_ridge, so the dense copy is short-lived.
    GloVe / Word2Vec are float32 .npy loaded with mmap_mode='r'.
4.  Weights are saved with np.savez_compressed (deflate on float32)
    and in the required .pkl model file along with selected alphas and
    X standardization statistics.
5.  gc.collect() after every major array release.
6.  Each (subject, embedding) run is fully self-contained; if one
    OOMs the others can still proceed.
"""

import os
import sys
import gc
import json
import logging
import pickle
import numpy as np
import scipy.sparse as sp

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from split import TRAIN_STORIES, TEST_STORIES
from preprocess import trim_Y
from ridge_utils.ridge import bootstrap_ridge

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)
_RESULTS = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
DM_DIR  = os.path.join(_RESULTS, 'design_matrices')
OUT_DIR = os.path.join(_RESULTS, 'ridge_31')
TMP_DIR = os.path.join(_RESULTS, 'tmp', 'ridge31')

for _d in (OUT_DIR, TMP_DIR):
    os.makedirs(_d, exist_ok=True)

# ── hyperparameters ──────────────────────────────────────────────────────────
SUBJECTS   = ['subject2', 'subject3']
EMBEDDINGS = ['bow', 'glove', 'word2vec']

ALPHAS   = np.logspace(1, 8, 20)   # 20 log-spaced alphas
NBOOTS   = 5                        # bootstrap samples for CV alpha selection
CHUNKLEN = 40                       # TR-chunk length for bootstrap holdout
NCHUNKS  = 5                        # # chunks held out per boot


# ════════════════════════════════════════════════════════════════════════════
#   X  helpers
# ════════════════════════════════════════════════════════════════════════════

def load_X_dense(emb: str, subject: str, split: str) -> np.ndarray:
    """Return a float32 dense array for the design matrix.

    Sparse .npz (BoW) → load CSR → .toarray()   (densified once, short-lived)
    Dense  .npy       → mmap then cast to float32
    """
    npz = os.path.join(DM_DIR, f'X_{emb}_{subject}_{split}.npz')
    npy = os.path.join(DM_DIR, f'X_{emb}_{subject}_{split}.npy')

    if os.path.exists(npz):
        log.info(f'    loading sparse  {os.path.basename(npz)}')
        X = sp.load_npz(npz).toarray().astype(np.float32)
    elif os.path.exists(npy):
        log.info(f'    loading dense   {os.path.basename(npy)}')
        # mmap_mode='r' avoids an immediate full copy; astype forces one, but
        # the mmap is freed right after so peak usage stays at 1× matrix size.
        X = np.array(np.load(npy, mmap_mode='r'), dtype=np.float32)
    else:
        raise FileNotFoundError(
            f'No design matrix for emb={emb} subject={subject} split={split}\n'
            f'  checked: {npz}\n           {npy}'
        )
    log.info(f'    X shape: {X.shape}  ({X.nbytes / 1e6:.1f} MB)')
    return X


def maybe_resave_bow_sparse(X: np.ndarray, emb: str, subject: str, split: str) -> None:
    """If BoW was saved as a dense .npy, convert and save as sparse .npz.
    Subsequent runs load the sparse version directly, saving disk + load time.
    """
    if emb != 'bow':
        return
    npz = os.path.join(DM_DIR, f'X_{emb}_{subject}_{split}.npz')
    if not os.path.exists(npz):
        Xs = sp.csr_matrix(X)
        sparsity = 1.0 - Xs.nnz / float(X.shape[0] * X.shape[1])
        log.info(f'    saving sparse BoW  sparsity={sparsity:.3%}  → {os.path.basename(npz)}')
        sp.save_npz(npz, Xs)


# ════════════════════════════════════════════════════════════════════════════
#   Y  helpers
# ════════════════════════════════════════════════════════════════════════════

def build_Y_memmap(stories: list, subject: str, tag: str) -> str:
    """Load Y story-by-story and write as float32 memmap.

    Memory cost during build ≈ max(Y_one_story)  ≪  Y_all_stories.
    Returns the path to the memmap file.
    """
    out_path = os.path.join(TMP_DIR, f'Y_{subject}_{tag}.npy')

    # first pass: measure dimensions
    n_rows, n_vox = 0, None
    for story in stories:
        Y  = np.load(os.path.join(DATA_DIR, subject, f'{story}.npy'), mmap_mode='r')
        Yt = trim_Y(Y)
        n_rows += Yt.shape[0]
        if n_vox is None:
            n_vox = Yt.shape[1]
        del Y, Yt

    if os.path.exists(out_path):
        cached = np.load(out_path, mmap_mode='r')
        if cached.shape == (n_rows, n_vox):
            log.info(f'    reusing cached memmap  {os.path.basename(out_path)}')
            del cached
            return out_path
        log.warning(
            f'    cached memmap shape {cached.shape} != expected {(n_rows, n_vox)}; rebuilding'
        )
        del cached

    log.info(f'    Y memmap  ({n_rows}, {n_vox})  '
             f'≈ {n_rows * n_vox * 4 / 1e9:.2f} GB')

    # second pass: write float32 memmap
    mm  = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32,
                                    shape=(n_rows, n_vox))
    row = 0
    for story in stories:
        Y  = np.load(os.path.join(DATA_DIR, subject, f'{story}.npy'), mmap_mode='r')
        Yt = trim_Y(Y).astype(np.float32)
        mm[row: row + Yt.shape[0]] = Yt
        row += Yt.shape[0]
        del Y, Yt
        gc.collect()

    del mm      # flush to disk
    gc.collect()
    return out_path


def zscore_memmap_inplace(mmap_path: str, vchunk: int = 2000) -> np.ndarray:
    """Z-score each column of a float32 memmap IN-PLACE.

    Processes [vchunk] voxels at a time so peak extra RAM is only
    n_TRs × vchunk × 8 bytes (float64 for numerical stability, then
    stored back as float32).  No second full copy of Y is ever created.
    """
    mm   = np.lib.format.open_memmap(mmap_path, mode='r+')
    T, V = mm.shape
    log.info(f'    z-scoring in-place  ({T} × {V})  vchunk={vchunk}')

    for v0 in range(0, V, vchunk):
        v1    = min(v0 + vchunk, V)
        chunk = mm[:, v0:v1].astype(np.float64)   # float64 for accuracy
        mu    = chunk.mean(0)
        sd    = chunk.std(0)
        sd[sd < 1e-10] = 1.0
        mm[:, v0:v1] = ((chunk - mu) / sd).astype(np.float32)
        del chunk

    del mm
    gc.collect()
    return np.load(mmap_path, mmap_mode='r')


# ════════════════════════════════════════════════════════════════════════════
#   Evaluation
# ════════════════════════════════════════════════════════════════════════════

def corr_stats(corrs: np.ndarray) -> dict:
    """
    Summarize voxel-wise held-out correlations for report tables.

    Mean/median describe whole-brain performance, while top-percentile
    summaries capture the upper tail of voxels that are most language
    responsive and therefore most relevant for interpretation.
    """
    return {
        'mean_cc':   float(np.mean(corrs)),
        'median_cc': float(np.median(corrs)),
        'top1_cc':   float(np.percentile(corrs, 99)),
        'top5_cc':   float(np.percentile(corrs, 95)),
        'n_positive': int((corrs > 0).sum()),
        'n_voxels':   int(len(corrs)),
        'frac_pos':   float((corrs > 0).mean()),
    }


# ════════════════════════════════════════════════════════════════════════════
#   Core run
# ════════════════════════════════════════════════════════════════════════════

def run(subject: str, emb: str) -> dict:
    log.info(f'\n{"="*64}')
    log.info(f'  Subject: {subject}   Embedding: {emb}')
    log.info(f'{"="*64}')

    # ── 1. Load X ────────────────────────────────────────────────────────
    log.info('  [1/5] Loading design matrices …')
    X_tr = load_X_dense(emb, subject, 'train')
    X_te = load_X_dense(emb, subject, 'test')
    maybe_resave_bow_sparse(X_tr, emb, subject, 'train')
    maybe_resave_bow_sparse(X_te, emb, subject, 'test')

    # ── 2. Build Y memmaps (writes to TMP_DIR) ───────────────────────────
    log.info('  [2/5] Building Y memmaps …')
    Y_tr_path = build_Y_memmap(TRAIN_STORIES, subject, f'{emb}_train')
    Y_te_path = build_Y_memmap(TEST_STORIES,  subject, f'{emb}_test')

    # ── 3. Z-score Y in-place (no second full copy) ──────────────────────
    log.info('  [3/5] Z-scoring Y in-place …')
    Y_tr = zscore_memmap_inplace(Y_tr_path)
    Y_te = zscore_memmap_inplace(Y_te_path)

    # ── 4. Z-score X, align lengths ──────────────────────────────────────
    log.info('  [4/5] Z-scoring X and aligning lengths …')
    # X standardization must be fit on training stories only. Reusing the
    # training mean/std on held-out stories avoids test-set leakage and keeps
    # all embedding methods on the same scale before ridge regression.
    x_mean = X_tr.mean(0, keepdims=True).astype(np.float32)
    x_std = X_tr.std(0, keepdims=True).astype(np.float32)
    x_std[x_std < 1e-10] = 1.0
    X_tr = ((X_tr - x_mean) / x_std).astype(np.float32)
    X_te = ((X_te - x_mean) / x_std).astype(np.float32)

    if X_tr.shape[0] != Y_tr.shape[0]:
        raise ValueError(
            f'length mismatch train X={X_tr.shape[0]} Y={Y_tr.shape[0]}; '
            'regenerate design matrices with the corrected preprocessing pipeline'
        )
    if X_te.shape[0] != Y_te.shape[0]:
        raise ValueError(
            f'length mismatch test X={X_te.shape[0]} Y={Y_te.shape[0]}; '
            'regenerate design matrices with the corrected preprocessing pipeline'
        )

    min_tr = X_tr.shape[0]
    min_te = X_te.shape[0]
    # Materialise the memmap slices into RAM as float32.
    # This is the one unavoidable full copy: bootstrap_ridge indexes Y
    # with arbitrary row indices that can't be served lazily from disk.
    log.info(f'    materialising Y  train={min_tr}×{Y_tr.shape[1]}  '
             f'test={min_te}×{Y_te.shape[1]} …')
    Y_tr_ram = np.array(Y_tr[:min_tr], dtype=np.float32)
    Y_te_ram = np.array(Y_te[:min_te], dtype=np.float32)
    del Y_tr, Y_te
    gc.collect()

    log.info(f'    X_train {X_tr.shape}   Y_train {Y_tr_ram.shape}')
    log.info(f'    X_test  {X_te.shape}   Y_test  {Y_te_ram.shape}')

    # ── 5. Bootstrap ridge ───────────────────────────────────────────────
    log.info(f'  [5/5] bootstrap_ridge  nboots={NBOOTS}  chunklen={CHUNKLEN} …')
    # bootstrap_ridge selects alpha voxel-by-voxel using held-out chunks from
    # the training stories, then evaluates the final weights on the fixed
    # held-out test stories passed as Pstim/Presp.
    wt, corrs, valphas, allRcorrs, valinds = bootstrap_ridge(
        Rstim     = X_tr,
        Rresp     = Y_tr_ram,
        Pstim     = X_te,
        Presp     = Y_te_ram,
        alphas    = ALPHAS,
        nboots    = NBOOTS,
        chunklen  = CHUNKLEN,
        nchunks   = NCHUNKS,
        use_corr  = True,
        return_wt = True,
    )

    corrs = np.array(corrs, dtype=np.float32)
    stats = corr_stats(corrs)

    log.info(
        f'  → mean CC={stats["mean_cc"]:.4f}  '
        f'median={stats["median_cc"]:.4f}  '
        f'top1%={stats["top1_cc"]:.4f}  '
        f'top5%={stats["top5_cc"]:.4f}  '
        f'pos_voxels={stats["n_positive"]}/{stats["n_voxels"]}'
    )

    # ── Save ─────────────────────────────────────────────────────────────
    prefix = os.path.join(OUT_DIR, f'{emb}_{subject}')

    # Weights: compressed .npz  (deflate; typically 3–5× smaller than raw)
    np.savez_compressed(
        f'{prefix}_weights.npz',
        weights = wt.astype(np.float32),
        valphas = valphas.astype(np.float32),
    )

    # Per-voxel correlations (float32 .npy)
    np.save(f'{prefix}_corrs.npy',   corrs)
    np.save(f'{prefix}_valphas.npy', valphas.astype(np.float32))

    # Bootstrap correlation cube (A×V×B) – save compressed
    if allRcorrs is not None:
        arc = np.array(allRcorrs, dtype=np.float32)
        np.savez_compressed(f'{prefix}_boot_corrs.npz', boot_corrs=arc)
        del arc

    model = {
        'emb':      emb,
        'subject':  subject,
        'model_type': 'ridge_encoding_model',
        'weights':  wt.astype(np.float32),
        'valphas':  valphas.astype(np.float32),
        'x_mean':   x_mean.astype(np.float32),
        'x_std':    x_std.astype(np.float32),
        'alphas':   ALPHAS.tolist(),
        'nboots':   NBOOTS,
        'chunklen': CHUNKLEN,
        'nchunks':  NCHUNKS,
        'stats':    stats,
        'X_train_shape': list(X_tr.shape),
        'Y_train_shape': list(Y_tr_ram.shape),
        'note': 'Predict z-scored fMRI responses from X standardized with x_mean/x_std.',
    }
    with open(f'{prefix}_model.pkl', 'wb') as fh:
        pickle.dump(model, fh, protocol=4)

    with open(f'{prefix}_stats.json', 'w') as fh:
        json.dump(stats, fh, indent=2)

    log.info(f'  Saved  {prefix}_*')

    # ── Cleanup ──────────────────────────────────────────────────────────
    del X_tr, X_te, Y_tr_ram, Y_te_ram, wt, corrs, valphas, allRcorrs, x_mean, x_std
    gc.collect()

    return stats


# ════════════════════════════════════════════════════════════════════════════
#   Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    all_results: dict = {}

    for subject in SUBJECTS:
        all_results[subject] = {}
        for emb in EMBEDDINGS:
            try:
                stats = run(subject, emb)
                all_results[subject][emb] = stats
            except FileNotFoundError as exc:
                log.warning(f'  SKIP {subject}/{emb}: {exc}')
            except MemoryError as exc:
                log.error(f'  OOM  {subject}/{emb}: {exc}')
            finally:
                gc.collect()

    # ── Summary table ────────────────────────────────────────────────────
    log.info('\n\n' + '='*72)
    log.info('SUMMARY')
    log.info('='*72)
    log.info(
        f'{"Subject":<12} {"Embedding":<12} '
        f'{"Mean CC":>10} {"Median CC":>10} {"Top 1%":>8} {"Top 5%":>8}'
    )
    log.info('-'*72)
    for subj, embs in all_results.items():
        for emb, s in embs.items():
            log.info(
                f'{subj:<12} {emb:<12} '
                f'{s["mean_cc"]:>10.4f} {s["median_cc"]:>10.4f} '
                f'{s["top1_cc"]:>8.4f} {s["top5_cc"]:>8.4f}'
            )

    with open(os.path.join(OUT_DIR, 'all_results.pkl'), 'wb') as fh:
        pickle.dump(all_results, fh, protocol=4)
    with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as fh:
        json.dump(all_results, fh, indent=2)

    log.info(f'\nAll results saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
