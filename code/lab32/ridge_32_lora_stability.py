"""
ridge_32_lora_stability.py  -  Ridge for LoRA rank stability embeddings
=======================================================================
Identical bootstrap CV pipeline to ridge_32.py, but evaluates only the
LoRA rank-stability embeddings and saves outputs separately.

Run from repo root:
    python code/lab32/ridge_32_lora_stability.py
"""

import os
import sys
import gc
import json
import logging
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab31.split import TRAIN_STORIES, TEST_STORIES
from lab31.preprocess import trim_Y
from ridge_utils.ridge import bootstrap_ridge


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)
RESULTS_DIR = os.environ.get(
    'RESULTS_DIR',
    os.path.join('/ocean/projects/mth250011p', os.environ.get('USER'), 'results'),
)
OUT_DIR = os.path.join(RESULTS_DIR, 'ridge_32_lora_stability')
TMP_DIR = os.path.join(RESULTS_DIR, 'tmp', 'ridge32')
DM_BASE = os.path.join(RESULTS_DIR, 'design_matrices')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

SUBJECTS = ['subject2', 'subject3']
EMBEDDINGS = ['bert_lora_r4', 'bert_lora_r16']

ALPHAS = np.logspace(1, 8, 20)
NBOOTS = 5
CHUNKLEN = 40
NCHUNKS = 5


def load_X_dense(emb: str, subject: str, split: str) -> np.ndarray:
    """Load one dense LoRA-rank stability design matrix."""
    subj_short = subject.replace('subject', 'subj')
    path = os.path.join(DM_BASE, f'X_{emb}_{subj_short}_{split}.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing design matrix: {path}')
    log.info(f'    loading {os.path.basename(path)}')
    X = np.array(np.load(path, mmap_mode='r'), dtype=np.float32)
    log.info(f'    X shape: {X.shape}  ({X.nbytes / 1e6:.1f} MB)')
    return X


def build_Y_memmap(stories: list, subject: str, tag: str) -> str:
    """Stack trimmed BOLD responses into a reusable on-disk memmap."""
    out_path = os.path.join(TMP_DIR, f'Y_{subject}_{tag}.npy')

    n_rows, n_vox = 0, None
    for story in stories:
        Y = np.load(os.path.join(DATA_DIR, subject, f'{story}.npy'), mmap_mode='r')
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

    log.info(f'    Y memmap  ({n_rows}, {n_vox})')
    mm = np.lib.format.open_memmap(
        out_path,
        mode='w+',
        dtype=np.float32,
        shape=(n_rows, n_vox),
    )
    row = 0
    for story in stories:
        Y = np.load(os.path.join(DATA_DIR, subject, f'{story}.npy'), mmap_mode='r')
        Yt = trim_Y(Y).astype(np.float32)
        mm[row: row + Yt.shape[0]] = Yt
        row += Yt.shape[0]
        del Y, Yt
        gc.collect()

    del mm
    gc.collect()
    return out_path


def zscore_memmap_inplace(mmap_path: str, vchunk: int = 2000) -> np.ndarray:
    """Z-score each voxel response in chunks to control memory use."""
    mm = np.lib.format.open_memmap(mmap_path, mode='r+')
    T, V = mm.shape
    log.info(f'    z-scoring in-place  ({T} x {V})  vchunk={vchunk}')

    for v0 in range(0, V, vchunk):
        v1 = min(v0 + vchunk, V)
        chunk = mm[:, v0:v1].astype(np.float64)
        mu = chunk.mean(0)
        sd = chunk.std(0)
        sd[sd < 1e-10] = 1.0
        mm[:, v0:v1] = ((chunk - mu) / sd).astype(np.float32)
        del chunk

    del mm
    gc.collect()
    return np.load(mmap_path, mmap_mode='r')


def corr_stats(corrs: np.ndarray) -> dict:
    """Summarize voxel-wise held-out CCs for the LoRA-rank comparison."""
    return {
        'mean_cc': float(np.mean(corrs)),
        'median_cc': float(np.median(corrs)),
        'top1_cc': float(np.percentile(corrs, 99)),
        'top5_cc': float(np.percentile(corrs, 95)),
        'n_positive': int((corrs > 0).sum()),
        'n_voxels': int(len(corrs)),
        'frac_pos': float((corrs > 0).mean()),
    }


def run(subject: str, emb: str) -> dict:
    log.info(f'\n{"=" * 64}')
    log.info(f'  Subject: {subject}   Embedding: {emb}')
    log.info(f'{"=" * 64}')

    log.info('  [1/5] Loading design matrices ...')
    X_tr = load_X_dense(emb, subject, 'train')
    X_te = load_X_dense(emb, subject, 'test')

    log.info('  [2/5] Building Y memmaps ...')
    Y_tr_path = build_Y_memmap(TRAIN_STORIES, subject, f'{subject}_train')
    Y_te_path = build_Y_memmap(TEST_STORIES, subject, f'{subject}_test')

    log.info('  [3/5] Z-scoring Y in-place ...')
    Y_tr = zscore_memmap_inplace(Y_tr_path)
    Y_te = zscore_memmap_inplace(Y_te_path)

    log.info('  [4/5] Z-scoring X and aligning lengths ...')
    # Use train-story feature statistics for both train and test splits, matching
    # the main ridge run and avoiding leakage from held-out stories.
    x_mean = X_tr.mean(0, keepdims=True).astype(np.float32)
    x_std = X_tr.std(0, keepdims=True).astype(np.float32)
    x_std[x_std < 1e-10] = 1.0
    X_tr = ((X_tr - x_mean) / x_std).astype(np.float32)
    X_te = ((X_te - x_mean) / x_std).astype(np.float32)

    if X_tr.shape[0] != Y_tr.shape[0]:
        raise ValueError(
            f'length mismatch train X={X_tr.shape[0]} Y={Y_tr.shape[0]}; '
            'regenerate LoRA stability design matrices with corrected preprocessing'
        )
    if X_te.shape[0] != Y_te.shape[0]:
        raise ValueError(
            f'length mismatch test X={X_te.shape[0]} Y={Y_te.shape[0]}; '
            'regenerate LoRA stability design matrices with corrected preprocessing'
        )

    min_tr = X_tr.shape[0]
    min_te = X_te.shape[0]
    Y_tr_ram = np.array(Y_tr[:min_tr], dtype=np.float32)
    Y_te_ram = np.array(Y_te[:min_te], dtype=np.float32)
    del Y_tr, Y_te
    gc.collect()

    log.info(f'    X_train {X_tr.shape}   Y_train {Y_tr_ram.shape}')
    log.info(f'    X_test  {X_te.shape}   Y_test  {Y_te_ram.shape}')

    log.info(f'  [5/5] bootstrap_ridge  nboots={NBOOTS}  chunklen={CHUNKLEN} ...')
    # The ridge procedure is intentionally unchanged from the main LoRA run so
    # any differences can be attributed to adapter rank rather than evaluation.
    wt, corrs, valphas, allRcorrs, valinds = bootstrap_ridge(
        Rstim=X_tr,
        Rresp=Y_tr_ram,
        Pstim=X_te,
        Presp=Y_te_ram,
        alphas=ALPHAS,
        nboots=NBOOTS,
        chunklen=CHUNKLEN,
        nchunks=NCHUNKS,
        use_corr=True,
        return_wt=True,
    )

    corrs = np.array(corrs, dtype=np.float32)
    stats = corr_stats(corrs)
    log.info(
        f'  -> mean CC={stats["mean_cc"]:.4f}  '
        f'median={stats["median_cc"]:.4f}  '
        f'top1%={stats["top1_cc"]:.4f}  '
        f'top5%={stats["top5_cc"]:.4f}'
    )

    prefix = os.path.join(OUT_DIR, f'{emb}_{subject}')
    np.savez_compressed(
        f'{prefix}_weights.npz',
        weights=wt.astype(np.float32),
        valphas=valphas.astype(np.float32),
    )
    np.save(f'{prefix}_corrs.npy', corrs)
    np.save(f'{prefix}_valphas.npy', valphas.astype(np.float32))

    if allRcorrs is not None:
        np.savez_compressed(
            f'{prefix}_boot_corrs.npz',
            boot_corrs=np.array(allRcorrs, dtype=np.float32),
        )

    model = {
        'emb': emb,
        'subject': subject,
        'model_type': 'ridge_encoding_model',
        'weights': wt.astype(np.float32),
        'valphas': valphas.astype(np.float32),
        'x_mean': x_mean.astype(np.float32),
        'x_std': x_std.astype(np.float32),
        'alphas': ALPHAS.tolist(),
        'nboots': NBOOTS,
        'chunklen': CHUNKLEN,
        'nchunks': NCHUNKS,
        'stats': stats,
        'X_train_shape': list(X_tr.shape),
        'Y_train_shape': list(Y_tr_ram.shape),
        'note': 'Predict z-scored fMRI responses from X standardized with x_mean/x_std.',
    }
    with open(f'{prefix}_model.pkl', 'wb') as fh:
        pickle.dump(model, fh, protocol=4)
    with open(f'{prefix}_stats.json', 'w') as fh:
        json.dump(stats, fh, indent=2)

    log.info(f'  Saved  {prefix}_*')
    del X_tr, X_te, Y_tr_ram, Y_te_ram, wt, corrs, valphas, allRcorrs, x_mean, x_std
    gc.collect()
    return stats


def main():
    all_results = {}
    for subject in SUBJECTS:
        all_results[subject] = {}
        for emb in EMBEDDINGS:
            try:
                all_results[subject][emb] = run(subject, emb)
            except FileNotFoundError as exc:
                log.warning(f'  SKIP {subject}/{emb}: {exc}')
            finally:
                gc.collect()

    log.info('\n\n' + '=' * 72)
    log.info('SUMMARY')
    log.info('=' * 72)
    for subject, embs in all_results.items():
        for emb, stats in embs.items():
            log.info(
                f'{subject:<12} {emb:<20} '
                f'{stats["mean_cc"]:>10.4f} {stats["median_cc"]:>10.4f} '
                f'{stats["top1_cc"]:>8.4f} {stats["top5_cc"]:>8.4f}'
            )

    with open(os.path.join(OUT_DIR, 'all_results.pkl'), 'wb') as fh:
        pickle.dump(all_results, fh, protocol=4)
    with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as fh:
        json.dump(all_results, fh, indent=2)
    log.info(f'\nAll results saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
