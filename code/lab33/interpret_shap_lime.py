"""
PNG-only SHAP and LIME wrapper interpretation for Lab 3.3.

For each selected test story and high-performing voxel, this script:
  1. perturbs story words and reruns BERT -> downsample -> trim/lag -> ridge,
  2. uses shap.KernelExplainer for SHAP word attributions,
  3. uses a LIME-style weighted local ridge surrogate,
  4. saves report-ready PNG plots.

The scalar explained by SHAP/LIME is the selected voxel's story-level held-out
correlation coefficient after a word perturbation, not the raw BOLD value at a
single TR. This keeps the interpretation tied to predictive performance.
"""

import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizerFast

try:
    import shap
except ImportError as exc:
    raise ImportError(
        'The shap package is required. Install/load it in the lab environment '
        'before running this script.'
    ) from exc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lab31.preprocess import get_downsampled, apply_trim_and_lag, align_X_to_Y_lengths, trim_Y


DATA_DIR = os.environ.get(
    'DATA_DIR',
    '/ocean/projects/mth250011p/shared/215a/final_project/data',
)
RESULTS_DIR = os.environ.get(
    'RESULTS_DIR',
    f'/ocean/projects/mth250011p/{os.environ.get("USER")}/results',
)
RIDGE_DIR = os.path.join(RESULTS_DIR, 'ridge_32')
DM_DIR = os.path.join(RESULTS_DIR, 'design_matrices')
MODEL_SOURCES = {
    'bert_pretrained': 'google-bert/bert-base-uncased',
    'bert_finetuned': os.path.join(RESULTS_DIR, 'models', 'bert_finetuned_checkpoint'),
}
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', 'lab33')
os.makedirs(FIG_DIR, exist_ok=True)

SUBJECT = os.environ.get('SUBJECT', 'subject3')
SUBJECTS = [
    s.strip()
    for s in os.environ.get('SUBJECTS', SUBJECT).split(',')
    if s.strip()
]
EMB = os.environ.get('EMB', 'bert_pretrained')
DEFAULT_TEST_STORIES = [
    'canplanetearthfeedtenbillionpeoplepart1',
    'stumblinginthedark',
]
TEST_STORIES = [
    s.strip()
    for s in os.environ.get('TEST_STORIES', ','.join(DEFAULT_TEST_STORIES)).split(',')
    if s.strip()
]

N_TOP_VOXELS = 3
MAX_CANDIDATE_WORDS = 60
N_LIME_PERTURB = 100
N_SHAP_SAMPLES = 120
KEEP_PROB = 0.75
KERNEL_WIDTH = 0.25
RIDGE_ALPHA_SURROGATE = 1e-3
SEED = 42
DEVICE = os.environ.get('DEVICE', 'cpu').lower()
if DEVICE == 'cuda' and not torch.cuda.is_available():
    print('Requested DEVICE=cuda, but CUDA is not available; falling back to CPU.')
    DEVICE = 'cpu'
SHAP_COLOR = '#D85A30'
LIME_COLOR = '#1D9E75'
NEG_COLOR = '#3A6EA5'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.12,
})


def load_weights() -> np.ndarray:
    """Load ridge weights for the current embedding and subject."""
    path = os.path.join(RIDGE_DIR, f'{EMB}_{SUBJECT}_weights.npz')
    return np.load(path)['weights'].astype(np.float32)


def load_train_stats() -> tuple[np.ndarray, np.ndarray]:
    """
    Recompute training-set feature scaling used by the ridge model.

    Perturbed stories must be standardized with the original training mean/std
    so SHAP/LIME scores are evaluated in the same feature space as ridge.
    """
    subj_short = SUBJECT.replace('subject', 'subj')
    X_train = np.load(
        os.path.join(DM_DIR, f'X_{EMB}_{subj_short}_train.npy'),
        mmap_mode='r',
    ).astype(np.float32)
    mu = X_train.mean(0, keepdims=True)
    sd = X_train.std(0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def corr_by_voxel(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Compute Pearson CC independently for each voxel column."""
    Y_true = Y_true.astype(np.float64)
    Y_pred = Y_pred.astype(np.float64)
    Y_true = Y_true - Y_true.mean(axis=0, keepdims=True)
    Y_pred = Y_pred - Y_pred.mean(axis=0, keepdims=True)
    num = np.sum(Y_true * Y_pred, axis=0)
    den = np.sqrt(np.sum(Y_true ** 2, axis=0) * np.sum(Y_pred ** 2, axis=0))
    out = np.zeros(Y_true.shape[1], dtype=np.float32)
    valid = den > 1e-10
    out[valid] = (num[valid] / den[valid]).astype(np.float32)
    return out


def embed_words(words: list[str], model, tokenizer, chunk_size=96, stride=48) -> np.ndarray:
    """
    Re-embed a perturbed word sequence with contextual BERT.

    This mirrors the Lab 3.2 contextual extraction: words are processed in
    overlapping chunks, subword hidden states are pooled back to word level, and
    words seen in multiple chunks are averaged.
    """
    model.eval()
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
            word_ids = inputs.word_ids(batch_index=0)
            inputs = inputs.to(DEVICE)

            hidden = model(**inputs).last_hidden_state[0]
            # Fast-tokenizer word IDs tell us which subword pieces belong to
            # each original word, preserving word-level attribution.
            for local_idx, original_idx in enumerate(valid_positions):
                token_positions = [
                    tok_idx for tok_idx, word_id in enumerate(word_ids)
                    if word_id == local_idx
                ]
                if token_positions:
                    pooled = hidden[token_positions].mean(dim=0)
                    word_idx = start + original_idx
                    embedding_sums[word_idx] += pooled.cpu().numpy()
                    embedding_counts[word_idx] += 1.0

    nonzero = embedding_counts > 0
    embeddings = np.zeros_like(embedding_sums)
    # Average overlapping contextual views so chunk boundaries do not dominate
    # the explanation for words near a window edge.
    embeddings[nonzero] = embedding_sums[nonzero] / embedding_counts[nonzero, None]
    return embeddings


def story_X_from_words(
    story: str,
    words: list[str],
    wordseqs: dict,
    model,
    tokenizer,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """
    Convert a perturbed story back into the ridge-ready design matrix.

    This is the core wrapper step: SHAP/LIME perturb raw words, but the encoding
    model consumes standardized, downsampled, HRF-delayed BERT features.
    """
    vectors = embed_words(words, model, tokenizer)
    X_down = get_downsampled([story], {story: vectors}, wordseqs)
    X_aligned = align_X_to_Y_lengths(X_down, SUBJECT, DATA_DIR)
    X_proc = apply_trim_and_lag(X_aligned)[story].astype(np.float32)
    return ((X_proc - mu) / sd).astype(np.float32)


def select_story_voxels(
    story: str,
    X_story: np.ndarray,
    weights: np.ndarray,
    n_top: int = N_TOP_VOXELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the top story-specific voxels by held-out CC for interpretation."""
    Y = trim_Y(np.load(os.path.join(DATA_DIR, SUBJECT, f'{story}.npy'), mmap_mode='r')).astype(np.float32)
    n = min(len(X_story), len(Y))
    Y = Y[:n]
    Y_pred = X_story[:n] @ weights
    corrs = corr_by_voxel(Y, Y_pred)
    voxels = np.argsort(corrs)[::-1][:n_top]
    print(f'\nTop {n_top} voxels for {story}:')
    for voxel in voxels:
        print(f'  voxel {voxel}: story CC={corrs[voxel]:.4f}')
    return voxels, corrs, Y


def candidate_positions(words: list[str], max_words: int = MAX_CANDIDATE_WORDS) -> np.ndarray:
    """
    Choose a manageable set of word positions for perturbation.

    Long stories are downsampled evenly to control the number of expensive
    BERT/ridge reruns required by SHAP and LIME.
    """
    valid = np.array([i for i, w in enumerate(words) if w and w != '<empty>'])
    if len(valid) <= max_words:
        return valid
    idx = np.linspace(0, len(valid) - 1, max_words).round().astype(int)
    return valid[idx]


def apply_mask(base_words: list[str], cand_pos: np.ndarray, z: np.ndarray, fill: str) -> list[str]:
    """Apply a binary keep/drop mask to candidate words using the requested fill."""
    perturbed = list(base_words)
    for pos in cand_pos[z == 0]:
        perturbed[pos] = fill
    return perturbed


def make_score_fn(
    story: str,
    base_words: list[str],
    cand_pos: np.ndarray,
    wordseqs: dict,
    model,
    tokenizer,
    weights: np.ndarray,
    voxel: int,
    y_true: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    fill: str,
):
    """
    Build the scalar function explained by SHAP/LIME for one voxel.

    The binary vector z indicates which candidate words remain. The returned
    score is the story-level CC for this voxel after rerunning the full encoding
    pipeline. A small cache avoids recomputing duplicate perturbation masks.
    """
    cache: dict[bytes, float] = {}

    def score_one(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=np.int8)
        key = z.tobytes()
        if key in cache:
            return cache[key]
        words = apply_mask(base_words, cand_pos, z, fill)
        X_pert = story_X_from_words(story, words, wordseqs, model, tokenizer, mu, sd)
        n = min(len(X_pert), len(y_true))
        y_pred = X_pert[:n] @ weights[:, voxel]
        score = float(corr_by_voxel(y_true[:n, [voxel]], y_pred[:, None])[0])
        cache[key] = score
        return score

    def score_many(Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z[None, :]
        scores = np.array([score_one(z) for z in Z], dtype=np.float64)
        print(f'    evaluated {len(scores)} perturbations ({len(cache)} unique cached)')
        return scores

    return score_many


def run_shap(score_fn, n_features: int) -> np.ndarray:
    """Run Kernel SHAP on the binary word-perturbation wrapper function."""
    background = np.zeros((1, n_features), dtype=np.float64)
    target = np.ones((1, n_features), dtype=np.float64)
    explainer = shap.KernelExplainer(score_fn, background)
    values = explainer.shap_values(target, nsamples=N_SHAP_SAMPLES)
    values = np.asarray(values)
    return values.reshape(-1).astype(np.float32)


def lime_kernel_weights(Z: np.ndarray) -> np.ndarray:
    """LIME exponential kernel based on the fraction of dropped words."""
    distances = 1.0 - Z.mean(axis=1)
    return np.exp(-(distances ** 2) / (KERNEL_WIDTH ** 2)).astype(np.float64)


def fit_weighted_surrogate(Z: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fit the weighted local ridge surrogate used as the LIME approximation."""
    X = np.column_stack([np.ones(len(Z)), Z.astype(np.float64)])
    sample_weights = lime_kernel_weights(Z)
    Xw = X * np.sqrt(sample_weights[:, None])
    yw = scores.astype(np.float64) * np.sqrt(sample_weights)
    penalty = RIDGE_ALPHA_SURROGATE * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(Xw.T @ Xw + penalty, Xw.T @ yw)
    return coef[1:].astype(np.float32)


def run_lime(score_fn, n_features: int, seed: int) -> np.ndarray:
    """Sample word-drop perturbations and return local surrogate coefficients."""
    rng = np.random.default_rng(seed)
    Z = np.ones((N_LIME_PERTURB + 1, n_features), dtype=np.int8)
    for i in range(1, N_LIME_PERTURB + 1):
        keep = rng.random(n_features) < KEEP_PROB
        if keep.all():
            keep[rng.integers(0, n_features)] = False
        Z[i] = keep.astype(np.int8)
    scores = score_fn(Z).astype(np.float32)
    print(f'    LIME base score={scores[0]:.4f}; perturb mean={scores[1:].mean():.4f}')
    return fit_weighted_surrogate(Z, scores)


def normalize_abs(values: np.ndarray) -> np.ndarray:
    """Scale absolute importances to [0, 1] within one voxel for plotting."""
    denom = float(np.max(np.abs(values))) if len(values) else 1.0
    if denom < 1e-12:
        denom = 1.0
    return np.abs(values) / denom


def top_union(shap_values: np.ndarray, lime_values: np.ndarray, n_top: int) -> np.ndarray:
    """Select words that are highly ranked by either SHAP or LIME."""
    score = np.maximum(np.abs(shap_values), np.abs(lime_values))
    return np.argsort(score)[::-1][:n_top]


def short_label(label: str, max_len: int = 14) -> str:
    """Shorten long word labels so figure axes remain readable."""
    label = str(label)
    return label if len(label) <= max_len else label[:max_len - 1] + '…'


def subject_label(subject: str) -> str:
    return subject.replace('subject', 'Subject ')


def story_label(story: str) -> str:
    if len(story) <= 28:
        return story
    return story[:25] + '...'


def save_fig(fig, name: str) -> None:
    """Save one Lab 3.3 report figure."""
    out = os.path.join(FIG_DIR, f'{name}.png')
    fig.savefig(out)
    print(f'  Saved: {out}')
    plt.close(fig)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(1, len(values) + 1)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def best_record(records: list[dict], story: str, subject: str) -> dict | None:
    candidates = [r for r in records if r['story'] == story and r['subject'] == subject]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r['cc'])


def plot_fig1_dual_bars(records: list[dict]) -> None:
    for story_num, story in enumerate(TEST_STORIES, start=1):
        story_records = [
            r for r in records
            if r['story'] == story and r['subject'] in SUBJECTS
        ]
        story_records = sorted(
            story_records,
            key=lambda r: (SUBJECTS.index(r['subject']), -r['cc']),
        )
        if not story_records:
            continue

        n_subjects = max(1, len(SUBJECTS))
        n_cols = max(
            len([r for r in story_records if r['subject'] == subject])
            for subject in SUBJECTS
        )
        fig, axes = plt.subplots(
            n_subjects, n_cols, figsize=(5.2 * n_cols, 4.8 * n_subjects),
            sharex=True, constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(n_subjects, n_cols)

        for row, subject in enumerate(SUBJECTS):
            subject_records = [r for r in story_records if r['subject'] == subject]
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(subject_records):
                    ax.set_visible(False)
                    continue
                record = subject_records[col]
                shap_abs_all = normalize_abs(record['shap_values'])
                lime_abs_all = normalize_abs(record['lime_values'])
                top = np.argsort(shap_abs_all)[::-1][:12][::-1]
                y = np.arange(len(top))

                ax.barh(y + 0.18, shap_abs_all[top], height=0.34, color=SHAP_COLOR, label='SHAP')
                ax.barh(y - 0.18, lime_abs_all[top], height=0.34, color=LIME_COLOR, label='LIME')
                ax.set_yticks(y)
                ax.set_yticklabels([short_label(record['labels'][i], 13) for i in top], fontsize=9)
                ax.tick_params(axis='y', labelsize=9)
                ax.set_xlim(0, 1.08)
                ax.set_xlabel('Absolute importance', fontsize=9)
                ax.set_title(
                    f"{subject_label(record['subject'])} | voxel {record['voxel']}\n"
                    f"CC={record['cc']:.3f}",
                    fontsize=10,
                )
                ax.legend(frameon=False, loc='lower right', fontsize=8)

        fig.suptitle(
            f'Top SHAP Words and Corresponding LIME Importance for Top {N_TOP_VOXELS} Voxels\n'
            f'{story_label(story)}',
            fontsize=14,
            fontweight='bold',
        )
        save_fig(fig, f'fig1_dual_importance_story{story_num}')


def plot_fig2_agreement(records: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(TEST_STORIES), figsize=(6.4 * len(TEST_STORIES), 5.8), constrained_layout=True)
    if len(TEST_STORIES) == 1:
        axes = [axes]

    for ax, story_num, story in zip(axes, range(1, len(TEST_STORIES) + 1), TEST_STORIES):
        story_records = [r for r in records if r['story'] == story]
        if not story_records:
            ax.set_visible(False)
            continue

        xs, ys, words, ranks_for_color = [], [], [], []
        for record in story_records:
            sx = normalize_abs(record['shap_values'])
            ly = normalize_abs(record['lime_values'])
            xs.extend(sx.tolist())
            ys.extend(ly.tolist())
            words.extend(record['labels'])
            ranks_for_color.extend(ranks(-sx).tolist())

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)
        rho = spearman(xs, ys)
        colors = np.array(ranks_for_color, dtype=np.float64)
        sc = ax.scatter(xs, ys, c=colors, cmap='viridis_r', s=30, alpha=0.72, edgecolors='none')
        lim = max(float(xs.max()), float(ys.max()), 1e-6)
        ax.plot([0, lim], [0, lim], color='black', linestyle='--', linewidth=1)

        top = np.argsort(xs)[::-1][:8]
        offsets = [(6, 6), (8, -10), (-34, 8), (-38, -10), (10, 14), (-40, 16), (12, -18), (-44, -18)]
        for rank_i, idx in enumerate(top):
            ax.annotate(
                short_label(words[idx], 12),
                (xs[idx], ys[idx]),
                xytext=offsets[rank_i % len(offsets)],
                textcoords='offset points',
                fontsize=8,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='0.82', alpha=0.9),
            )

        ax.set_title(f'{story_label(story)}\nSpearman rho={rho:.2f}', fontsize=11)
        ax.set_xlabel('SHAP absolute importance')
        ax.set_ylabel('LIME absolute importance')
        ax.set_xlim(-0.02, lim * 1.08)
        ax.set_ylim(-0.02, lim * 1.08)

    fig.colorbar(sc, ax=axes, shrink=0.86, label='SHAP importance rank')
    fig.suptitle('SHAP/LIME Agreement Across Selected Voxels', fontsize=14, fontweight='bold')
    save_fig(fig, 'fig2_shap_lime_agreement_by_story')


def plot_fig3_timelines(records: list[dict]) -> None:
    for story_num, story in enumerate(TEST_STORIES, start=1):
        story_records = [r for r in records if r['story'] == story]
        if not story_records:
            continue

        fig, axes = plt.subplots(
            len(story_records), 1, figsize=(12.5, 2.2 * len(story_records) + 1.1),
            sharex=True, constrained_layout=True,
        )
        if len(story_records) == 1:
            axes = [axes]

        max_abs = max(float(np.max(np.abs(r['shap_values']))) for r in story_records)
        max_abs = max(max_abs, 1e-9)
        for ax, record in zip(axes, story_records):
            values = record['shap_values']
            cand_pos = record['cand_pos']
            colors = np.where(values >= 0, SHAP_COLOR, NEG_COLOR)
            ax.vlines(cand_pos, 0, values, color=colors, alpha=0.78, linewidth=1.5)
            ax.scatter(cand_pos, values, c=colors, s=30, edgecolors='white', linewidths=0.5, zorder=3)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylim(-1.32 * max_abs, 1.32 * max_abs)
            ax.set_ylabel(f"{subject_label(record['subject'])}\nvoxel {record['voxel']}\nCC={record['cc']:.3f}", fontsize=8)

            threshold = np.percentile(np.abs(values), 90)
            for idx in np.where(np.abs(values) >= threshold)[0]:
                ax.annotate(
                    short_label(record['labels'][idx], 11),
                    (cand_pos[idx], values[idx]),
                    xytext=(0, 8 if values[idx] >= 0 else -13),
                    textcoords='offset points',
                    rotation=45,
                    ha='left',
                    va='bottom' if values[idx] >= 0 else 'top',
                    fontsize=7.2,
                )

        axes[-1].set_xlabel('Original word position in story')
        fig.suptitle(f'Where Important Words Occur\n{story_label(story)}', fontsize=14, fontweight='bold')
        save_fig(fig, f'fig3_timeline_story{story_num}')


def plot_fig4_heatmaps(records: list[dict]) -> None:
    for story_num, story in enumerate(TEST_STORIES, start=1):
        story_records = [r for r in records if r['story'] == story]
        if not story_records:
            continue

        score_by_word: dict[str, float] = {}
        for record in story_records:
            for label, shap_value, lime_value in zip(record['labels'], record['shap_values'], record['lime_values']):
                word = short_label(label, 18)
                score = max(abs(float(shap_value)), abs(float(lime_value)))
                score_by_word[word] = max(score_by_word.get(word, 0.0), score)

        top_words = [w for w, _ in sorted(score_by_word.items(), key=lambda kv: kv[1], reverse=True)[:20]]
        matrix = np.zeros((len(story_records), len(top_words)), dtype=np.float64)
        row_labels = []

        for row, record in enumerate(story_records):
            row_labels.append(f"{subject_label(record['subject'])} v{record['voxel']} CC={record['cc']:.2f}")
            values_by_word = {
                short_label(label, 18): float(value)
                for label, value in zip(record['labels'], record['shap_values'])
            }
            for col, word in enumerate(top_words):
                matrix[row, col] = values_by_word.get(word, 0.0)

        col_mean = matrix.mean(axis=0, keepdims=True)
        col_sd = matrix.std(axis=0, keepdims=True)
        col_sd[col_sd < 1e-9] = 1.0
        z = (matrix - col_mean) / col_sd

        fig, ax = plt.subplots(figsize=(12.6, 0.55 * len(story_records) + 4.3), constrained_layout=True)
        im = ax.imshow(z, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xticks(np.arange(len(top_words)))
        ax.set_xticklabels(top_words, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'Shared and Subject-Specific Word Effects\n{story_label(story)}', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.86, label='Column-z-scored signed SHAP')
        save_fig(fig, f'fig4_heatmap_story{story_num}')


def plot_appendix_grid(records: list[dict]) -> None:
    if not records:
        return

    fig, axes = plt.subplots(
        len(TEST_STORIES), len(SUBJECTS) * N_TOP_VOXELS,
        figsize=(4.1 * len(SUBJECTS) * N_TOP_VOXELS, 4.2 * len(TEST_STORIES)),
        squeeze=False,
        constrained_layout=True,
    )
    max_abs = max(float(np.max(np.abs(r['shap_values']))) for r in records)
    max_abs = max(max_abs, 1e-9)

    for row, story in enumerate(TEST_STORIES):
        story_records = [r for r in records if r['story'] == story]
        story_records = sorted(story_records, key=lambda r: (r['subject'], -r['cc']))
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if col >= len(story_records):
                ax.set_visible(False)
                continue
            record = story_records[col]
            values = record['shap_values']
            colors = np.where(values >= 0, SHAP_COLOR, NEG_COLOR)
            ax.vlines(record['cand_pos'], 0, values, color=colors, alpha=0.75, linewidth=1.1)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylim(-1.25 * max_abs, 1.25 * max_abs)
            ax.set_title(
                f"{subject_label(record['subject'])} v{record['voxel']}\nCC={record['cc']:.3f}",
                fontsize=9,
            )
            if col == 0:
                ax.set_ylabel(story_label(story), fontsize=9)
            if row == len(TEST_STORIES) - 1:
                ax.set_xlabel('word position')

    fig.suptitle('SHAP Timelines for All Selected Voxels', fontsize=14, fontweight='bold')
    save_fig(fig, 'appendix_all_voxel_timelines')


def make_report_figures(records: list[dict]) -> None:
    if not records:
        print('No SHAP/LIME records were generated; skipping plots.')
        return
    plot_fig1_dual_bars(records)
    plot_fig2_agreement(records)
    plot_fig3_timelines(records)
    plot_fig4_heatmaps(records)
    plot_appendix_grid(records)


def main():
    global SUBJECT
    if EMB not in MODEL_SOURCES:
        raise ValueError(f'Unsupported EMB={EMB}. Choose one of: {", ".join(MODEL_SOURCES)}')

    print(f'Loading wordseqs from {DATA_DIR}...')
    with open(os.path.join(DATA_DIR, 'raw_text.pkl'), 'rb') as fh:
        wordseqs = pickle.load(fh)

    print(f'Embedding/model: {EMB}')
    print(f'Subjects: {SUBJECTS}')
    print(f'Test stories: {TEST_STORIES}')
    print(f'Loading tokenizer and BERT encoder on {DEVICE}...')
    tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')
    mlm_model = BertForMaskedLM.from_pretrained(MODEL_SOURCES[EMB]).to(DEVICE)
    model = mlm_model.bert

    records = []
    for subject in SUBJECTS:
        SUBJECT = subject
        print('\n' + '#' * 76)
        print(f'Subject: {SUBJECT}')
        print('#' * 76)

        weights = load_weights()
        mu, sd = load_train_stats()

        for story_num, story in enumerate(TEST_STORIES, start=1):
            print('\n' + '=' * 72)
            print(f'Story {story_num}: {story}')
            print('=' * 72)

            base_words = list(wordseqs[story].data)
            X_base = story_X_from_words(story, base_words, wordseqs, model, tokenizer, mu, sd)
            voxels, story_corrs, y_true = select_story_voxels(story, X_base, weights)
            cand_pos = candidate_positions(base_words)
            labels = [base_words[pos] if base_words[pos] else '<empty>' for pos in cand_pos]
            print(f'Using {len(cand_pos)} candidate word positions.')

            for voxel in voxels:
                print(f'\n  Interpreting voxel {voxel}...')
                shap_score_fn = make_score_fn(
                    story, base_words, cand_pos, wordseqs, model, tokenizer,
                    weights, voxel, y_true, mu, sd, tokenizer.mask_token,
                )
                lime_score_fn = make_score_fn(
                    story, base_words, cand_pos, wordseqs, model, tokenizer,
                    weights, voxel, y_true, mu, sd, '',
                )

                print('    Running actual SHAP KernelExplainer...')
                shap_values = run_shap(shap_score_fn, len(cand_pos))
                print('    Running LIME weighted local surrogate...')
                lime_values = run_lime(lime_score_fn, len(cand_pos), SEED + int(voxel))

                record = {
                    'subject': SUBJECT,
                    'story': story,
                    'story_num': story_num,
                    'voxel': int(voxel),
                    'cc': float(story_corrs[voxel]),
                    'cand_pos': cand_pos.astype(np.int32),
                    'labels': labels,
                    'shap_values': shap_values.astype(np.float32),
                    'lime_values': lime_values.astype(np.float32),
                }
                records.append(record)

        del weights, mu, sd

    print('\nCreating report figures...')
    make_report_figures(records)
    print('\nDone. Figures and SHAP/LIME values saved in figures/lab33.')


if __name__ == '__main__':
    main()
