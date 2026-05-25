"""
plot_31.py  –  Lab 3.1 Figures
================================
Run from repo root:
    python code/lab31/plot_31.py

Works with EITHER version of ridge_31.py:
  - New version: loads pre-computed _per_story.pkl directly
  - Old version (no pkl): recomputes per-story CC on-the-fly from
    saved weights + raw data. No re-training needed.

Generates (saved to figures/lab31/):
  fig1_cc_comparison.png      – mean/median/top1/top5 CC bar chart
  fig2_cc_distribution.png    – voxel CC histogram for the best embedding
  fig3_stability_stories.png  – per-story mean CC bar chart
  fig4_stability_subjects.png – cross-subject scatter + bar (if 2 subjects)
"""

import os
import sys
import json
import pickle
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from split import TRAIN_STORIES, TEST_STORIES
from preprocess import trim_Y

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)

_RESULTS  = os.environ.get('RESULTS_DIR', os.path.join(os.path.dirname(__file__), '..', '..', 'results'))
RIDGE_DIR = os.path.join(_RESULTS, 'ridge_31')
DM_DIR    = os.path.join(_RESULTS, 'design_matrices')
FIG_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', 'lab31')

os.makedirs(FIG_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
# Only include subjects that actually have results
def _available_subjects():
    found = []
    for s in ['subject2', 'subject3']:
        if any(
            os.path.exists(os.path.join(RIDGE_DIR, f'{e}_{s}_corrs.npy'))
            for e in ['bow', 'glove', 'word2vec']
        ):
            found.append(s)
    return found

SUBJECTS    = _available_subjects()
EMBEDDINGS  = ['bow', 'glove', 'word2vec']
EMB_LABELS  = {'bow': 'BoW', 'glove': 'GloVe', 'word2vec': 'Word2Vec'}
SUBJ_LABELS = {'subject2': 'Subject 2', 'subject3': 'Subject 3'}

EMB_COLORS  = {'bow': '#4878CF', 'glove': '#6ACC65', 'word2vec': '#D65F5F'}
SUBJ_COLORS = {'subject2': '#2C7BB6', 'subject3': '#D7191C'}

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.1,
})


# ════════════════════════════════════════════════════════════════════════════
#  Loaders
# ════════════════════════════════════════════════════════════════════════════

def load_corrs(subject, emb):
    """Load voxel-wise held-out CC values for one subject/embedding."""
    p = os.path.join(RIDGE_DIR, f'{emb}_{subject}_corrs.npy')
    return np.load(p) if os.path.exists(p) else None


def load_stats(subject, emb):
    """Load precomputed summary metrics saved by ridge_31.py."""
    p = os.path.join(RIDGE_DIR, f'{emb}_{subject}_stats.json')
    return json.load(open(p)) if os.path.exists(p) else None


def load_weights(subject, emb):
    """Load regression weights from compressed .npz."""
    p = os.path.join(RIDGE_DIR, f'{emb}_{subject}_weights.npz')
    if not os.path.exists(p):
        return None
    data = np.load(p)
    return data['weights'].astype(np.float32)   # (n_features, n_voxels)


def load_X_dense(emb, subject, split):
    """Load design matrix, preferring sparse .npz over dense .npy."""
    import scipy.sparse as sp
    npz = os.path.join(DM_DIR, f'X_{emb}_{subject}_{split}.npz')
    npy = os.path.join(DM_DIR, f'X_{emb}_{subject}_{split}.npy')
    if os.path.exists(npz):
        return sp.load_npz(npz).toarray().astype(np.float32)
    elif os.path.exists(npy):
        return np.array(np.load(npy, mmap_mode='r'), dtype=np.float32)
    return None


def best_embedding(subject):
    """
    Choose the embedding with the highest whole-brain mean CC.

    This is used only for exploratory Lab 3.1 plots where a single embedding
    must be selected for distributions or per-story views.
    """
    best_emb, best_cc = None, -np.inf
    for emb in EMBEDDINGS:
        s = load_stats(subject, emb)
        if s and s['mean_cc'] > best_cc:
            best_cc, best_emb = s['mean_cc'], emb
    return best_emb


# ════════════════════════════════════════════════════════════════════════════
#  Per-story CC  (load pre-computed pkl OR compute on-the-fly from weights)
# ════════════════════════════════════════════════════════════════════════════

def _compute_per_story_from_weights(subject, emb):
    """Compute per-story mean voxel CC using saved weights.

    Called when _per_story.pkl does not exist (old ridge_31.py run).
    Loads X_test story-by-story, predicts with wt, correlates with Y.
    Memory cost: max(X_one_story) + max(Y_one_story) at a time.
    """
    wt = load_weights(subject, emb)
    if wt is None:
        print(f'    no weights for {subject}/{emb}, skipping')
        return None

    # Z-score params from train X — must match ridge_31.py's zscore_train_test_X
    X_tr_full = load_X_dense(emb, subject, 'train')
    if X_tr_full is None:
        return None
    mu_x = X_tr_full.mean(0, keepdims=True).astype(np.float32)
    sd_x = X_tr_full.std(0,  keepdims=True).astype(np.float32)
    sd_x[sd_x < 1e-10] = 1.0
    del X_tr_full
    gc.collect()

    X_te_full = load_X_dense(emb, subject, 'test')
    if X_te_full is None:
        return None

    # Slice X per story (rows correspond to TEST_STORIES in order)
    row = 0
    story_slices = {}
    for story in TEST_STORIES:
        Y_raw  = np.load(os.path.join(DATA_DIR, subject, f'{story}.npy'), mmap_mode='r')
        n_trs  = trim_Y(Y_raw).shape[0]
        story_slices[story] = (row, row + n_trs)
        row += n_trs
        del Y_raw

    results = {}
    for story in TEST_STORIES:
        r0, r1 = story_slices[story]
        X_s = (X_te_full[r0:r1] - mu_x) / sd_x       # z-scored X slice

        Y_raw = np.load(os.path.join(DATA_DIR, subject, f'{story}.npy'), mmap_mode='r')
        Y_s   = trim_Y(Y_raw).astype(np.float32)
        n     = min(X_s.shape[0], Y_s.shape[0])
        X_s, Y_s = X_s[:n], Y_s[:n]

        # z-score Y
        mu_y  = Y_s.mean(0, keepdims=True)
        sd_y  = Y_s.std(0,  keepdims=True)
        sd_y[sd_y < 1e-10] = 1.0
        Y_s   = (Y_s - mu_y) / sd_y

        # predict + z-score prediction
        pred  = X_s @ wt
        mu_p  = pred.mean(0, keepdims=True)
        sd_p  = pred.std(0,  keepdims=True)
        sd_p[sd_p < 1e-10] = 1.0
        pred  = (pred - mu_p) / sd_p

        corrs = (Y_s * pred).mean(0).astype(np.float32)
        results[story] = corrs

        print(f'    {story:20s}  mean CC={corrs.mean():.4f}  top1%={np.percentile(corrs,99):.4f}')
        del Y_raw, Y_s, X_s, pred
        gc.collect()

    del X_te_full, wt
    gc.collect()
    return results


def get_per_story(subject, emb):
    """Return {story: corrs_array}.  Load pkl if it exists, else compute."""
    pkl = os.path.join(RIDGE_DIR, f'{emb}_{subject}_per_story.pkl')
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            return pickle.load(f)
    print(f'  No _per_story.pkl for {subject}/{emb} — computing from weights …')
    result = _compute_per_story_from_weights(subject, emb)
    if result is not None:
        with open(pkl, 'wb') as f:
            pickle.dump(result, f, protocol=4)
        print(f'  Cached → {os.path.basename(pkl)}')
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Fig 1 – CC comparison across embeddings & subjects
# ════════════════════════════════════════════════════════════════════════════

def fig1_cc_comparison():
    metrics       = ['mean_cc', 'median_cc', 'top5_cc', 'top1_cc']
    metric_labels = ['Mean CC', 'Median CC', 'Top 5% CC', 'Top 1% CC']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Ridge Regression – CC Across Embeddings', fontsize=13, fontweight='bold')

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        x     = np.arange(len(EMBEDDINGS))
        width = 0.8 / max(len(SUBJECTS), 1)

        for si, subject in enumerate(SUBJECTS):
            vals = [
                (load_stats(subject, emb) or {}).get(metric, 0.0)
                for emb in EMBEDDINGS
            ]
            offset = (si - len(SUBJECTS)/2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=SUBJ_LABELS[subject],
                color=SUBJ_COLORS[subject],
                alpha=0.85, edgecolor='white', linewidth=0.5,
            )
            for bar, v in zip(bars, vals):
                if v != 0:
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + ax.get_ylim()[1] * 0.01,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)

        ax.set_title(mlabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([EMB_LABELS[e] for e in EMBEDDINGS])
        ax.set_ylabel('Pearson r')
        ax.margins(y=0.2)
        if metric == 'mean_cc' and len(SUBJECTS) > 1:
            ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig1_cc_comparison.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════════════
#  Fig 2 – Voxel CC distribution for best embedding
# ════════════════════════════════════════════════════════════════════════════

def fig2_cc_distribution():
    n_subj = len(SUBJECTS)
    fig, axes = plt.subplots(1, n_subj, figsize=(6 * n_subj, 4.5))
    if n_subj == 1:
        axes = [axes]
    fig.suptitle('Distribution of Voxel-wise CC (Best Embedding)',
                 fontsize=13, fontweight='bold')

    for ax, subject in zip(axes, SUBJECTS):
        best  = best_embedding(subject)
        corrs = load_corrs(subject, best)
        if corrs is None:
            ax.set_title(f'{SUBJ_LABELS[subject]} – data missing')
            continue

        color = SUBJ_COLORS[subject]
        n, bins, _ = ax.hist(corrs, bins=80, color=color, alpha=0.72,
                             edgecolor='none', density=True)

        # KDE overlay
        kde_x = np.linspace(corrs.min(), corrs.max(), 400)
        kde   = stats.gaussian_kde(corrs, bw_method=0.15)
        ax.plot(kde_x, kde(kde_x), color=color, lw=2)

        # reference lines
        ax.axvline(0,    color='black',  lw=0.9, ls='--', label='r = 0')
        ax.axvline(0.10, color='orange', lw=1.2, ls='--', label='PCS threshold r = 0.10')
        ax.axvspan(0.10, corrs.max(), alpha=0.07, color='green')

        s = load_stats(subject, best)
        ax.set_title(
            f'{SUBJ_LABELS[subject]}  –  {EMB_LABELS[best]}\n'
            f'mean={s["mean_cc"]:.3f}  median={s["median_cc"]:.3f}  '
            f'top 1%={s["top1_cc"]:.3f}',
            fontsize=10
        )
        ax.set_xlabel('Pearson r (test set)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

        frac_pos = (corrs > 0).mean()
        frac_pcs = (corrs > 0.10).mean()
        ax.text(0.97, 0.95,
                f'{frac_pos*100:.1f}% voxels  r > 0\n'
                f'{frac_pcs*100:.1f}% voxels  r > 0.10',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_cc_distribution.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════════════
#  Fig 3 – Stability across test stories
# ════════════════════════════════════════════════════════════════════════════

def fig3_stability_stories():
    n_subj = len(SUBJECTS)
    fig, axes = plt.subplots(1, n_subj, figsize=(7 * n_subj, 5))
    if n_subj == 1:
        axes = [axes]
    fig.suptitle('Stability Across Test Stories (Best Embedding)', fontsize=13, fontweight='bold')

    for ax, subject in zip(axes, SUBJECTS):
        data_by_emb = {}
        story_names = None

        # Only compute per-story for the best embedding — avoids loading
        # the huge BoW weight matrix (~6 GB) when it's not the best model.
        emb_to_run = [best_embedding(subject)]
        print(f'  {subject}: best embedding = {emb_to_run[0]}')

        for emb in emb_to_run:
            per_story = get_per_story(subject, emb)
            if per_story is None:
                continue
            means = {s: float(v.mean()) for s, v in per_story.items()}
            sorted_stories = sorted(means.keys())
            if story_names is None:
                story_names = sorted_stories
            data_by_emb[emb] = [means[s] for s in story_names]

        if not data_by_emb or story_names is None:
            ax.set_title(f'{SUBJ_LABELS[subject]} – data missing')
            continue

        n_emb     = len(data_by_emb)
        n_stories = len(story_names)
        x         = np.arange(n_stories)
        width     = 0.8 / n_emb

        for i, (emb, vals) in enumerate(data_by_emb.items()):
            offset = (i - n_emb/2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=EMB_LABELS[emb], color=EMB_COLORS[emb],
                   alpha=0.85, edgecolor='white', linewidth=0.4)

        ax.axhline(0, color='black', lw=0.7, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels([s[:14] for s in story_names],
                           rotation=40, ha='right', fontsize=8)
        ax.set_title(SUBJ_LABELS[subject], fontsize=11)
        ax.set_ylabel('Mean Voxel CC')
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig3_stability_stories.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════════════
#  Fig 4 – Cross-subject stability (scatter + bar)
# ════════════════════════════════════════════════════════════════════════════

def fig4_stability_subjects():
    if len(SUBJECTS) < 2:
        print('  Skipping fig4: only one subject available.')
        return

    s1, s2 = SUBJECTS[0], SUBJECTS[1]
    fig    = plt.figure(figsize=(15, 4.5))
    fig.suptitle(
        f'Cross-Subject Stability  ({SUBJ_LABELS[s1]} vs {SUBJ_LABELS[s2]})',
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(1, len(EMBEDDINGS) + 1,
                           width_ratios=[1] * len(EMBEDDINGS) + [0.9])

    for i, emb in enumerate(EMBEDDINGS):
        ax = fig.add_subplot(gs[i])
        c1 = load_corrs(s1, emb)
        c2 = load_corrs(s2, emb)

        if c1 is None or c2 is None:
            ax.set_title(f'{EMB_LABELS[emb]}\n(data missing)')
            continue

        n      = min(len(c1), len(c2))
        c1, c2 = c1[:n], c2[:n]

        ax.hist2d(c1, c2, bins=80, cmap='Blues',
                  range=[[c1.min(), c1.max()], [c2.min(), c2.max()]])

        lim = [min(c1.min(), c2.min()), max(c1.max(), c2.max())]
        ax.plot(lim, lim, 'r--', lw=1, alpha=0.8, label='y = x')

        r, _ = stats.pearsonr(c1, c2)
        ax.set_title(f'{EMB_LABELS[emb]}\ninter-subject r = {r:.3f}', fontsize=10)
        ax.set_xlabel(SUBJ_LABELS[s1])
        if i == 0:
            ax.set_ylabel(SUBJ_LABELS[s2])
        ax.legend(fontsize=8)

    # Right panel: mean CC bar chart
    ax_bar = fig.add_subplot(gs[len(EMBEDDINGS)])
    x      = np.arange(len(EMBEDDINGS))
    width  = 0.35

    for si, subject in enumerate(SUBJECTS):
        means = [
            (load_stats(subject, emb) or {}).get('mean_cc', 0.0)
            for emb in EMBEDDINGS
        ]
        ax_bar.bar(x + (si - 0.5) * width, means, width,
                   label=SUBJ_LABELS[subject],
                   color=SUBJ_COLORS[subject], alpha=0.85, edgecolor='white')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([EMB_LABELS[e] for e in EMBEDDINGS],
                            rotation=20, ha='right')
    ax_bar.set_ylabel('Mean Voxel CC')
    ax_bar.set_title('Mean CC\nby Subject', fontsize=10)
    ax_bar.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig4_stability_subjects.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'  Saved: {out}')


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    print(f'Available subjects: {SUBJECTS}')
    print(f'Output dir: {FIG_DIR}\n')

    print('Fig 1: CC comparison …')
    fig1_cc_comparison()

    print('Fig 2: CC distribution …')
    fig2_cc_distribution()

    print('Fig 3: Stability across stories …')
    fig3_stability_stories()

    print('Fig 4: Cross-subject stability …')
    fig4_stability_subjects()

    print(f'\nDone. All figures in {FIG_DIR}')


if __name__ == '__main__':
    main()
