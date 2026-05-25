"""
plot_32_stability.py - Lab 3.2 PCS/stability figures
=====================================================

Run from repo root:
    python code/lab32/plot_32_stability.py

This script intentionally keeps stability/control models out of the main
Lab 3.2 plots. It compares:
  1. contextual pretrained BERT vs. noncontextual word-level BERT, and
  2. optional LoRA rank stability if r=4/r=16 ridge outputs exist.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RIDGE32_DIR = RESULTS_DIR / "ridge_32"
RIDGE32_STABILITY_DIR = RESULTS_DIR / "ridge_32_stability"
RIDGE32_LORA_STABILITY_DIR = RESULTS_DIR / "ridge_32_lora_stability"
METRIC_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "figures" / "lab32"

METRIC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["subject2", "subject3"]
SUBJECT_LABELS = {"subject2": "Subject 2", "subject3": "Subject 3"}
SUBJECT_MARKERS = {"subject2": "o", "subject3": "s"}

METRICS = ["mean_cc", "median_cc", "top5_cc", "top1_cc"]
METRIC_LABELS = {
    "mean_cc": "Mean CC",
    "median_cc": "Median CC",
    "top5_cc": "Top 5% CC",
    "top1_cc": "Top 1% CC",
}

CONTEXT_MODELS = {
    "bert_pretrained": {
        "label": "Contextual BERT",
        "dir": RIDGE32_DIR,
        "color": "#4C86BD",
    },
    "bert_pretrained_wordlevel": {
        "label": "Word-level BERT",
        "dir": RIDGE32_STABILITY_DIR,
        "color": "#8A70B8",
    },
}

LORA_RANK_MODELS = {
    4: {"embedding": "bert_lora_r4", "dir": RIDGE32_LORA_STABILITY_DIR, "color": "#89BDE0"},
    8: {"embedding": "bert_lora", "dir": RIDGE32_DIR, "color": "#5BAFA3"},
    16: {"embedding": "bert_lora_r16", "dir": RIDGE32_LORA_STABILITY_DIR, "color": "#366E6A"},
}

WELL_PREDICTED_THRESHOLD = 0.10

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def stats_path(model: str, subject: str) -> Path:
    """Path to aggregate CC metrics for the contextual-vs-wordlevel comparison."""
    spec = CONTEXT_MODELS[model]
    return spec["dir"] / f"{model}_{subject}_stats.json"


def corrs_path(model: str, subject: str) -> Path:
    """Path to voxel-wise CCs for the contextual-vs-wordlevel comparison."""
    spec = CONTEXT_MODELS[model]
    return spec["dir"] / f"{model}_{subject}_corrs.npy"


def load_stats(model: str, subject: str):
    """Load aggregate stability metrics; return None when the run has not finished."""
    path = stats_path(model, subject)
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def load_corrs(model: str, subject: str):
    """Memory-map voxel-wise correlations for histogram/boxplot stability figures."""
    path = corrs_path(model, subject)
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def lora_stats_path(rank: int, subject: str) -> Path:
    """Path to aggregate CC metrics for one LoRA-rank stability run."""
    spec = LORA_RANK_MODELS[rank]
    return spec["dir"] / f"{spec['embedding']}_{subject}_stats.json"


def lora_corrs_path(rank: int, subject: str) -> Path:
    """Path to voxel-wise CCs for one LoRA-rank stability run."""
    spec = LORA_RANK_MODELS[rank]
    return spec["dir"] / f"{spec['embedding']}_{subject}_corrs.npy"


def load_lora_stats(rank: int, subject: str):
    """Load LoRA-rank summary metrics when that rank has been evaluated."""
    path = lora_stats_path(rank, subject)
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def load_lora_corrs(rank: int, subject: str):
    """Memory-map LoRA-rank voxel-wise correlations for distribution plots."""
    path = lora_corrs_path(rank, subject)
    if not path.exists():
        return None
    return np.load(path, mmap_mode="r")


def available_context_models():
    """Return context-control models with at least one completed subject."""
    return [
        model for model in CONTEXT_MODELS
        if any(stats_path(model, subject).exists() for subject in SUBJECTS)
    ]


def available_lora_ranks():
    """Return LoRA ranks with at least one completed ridge result."""
    return [
        rank for rank in sorted(LORA_RANK_MODELS)
        if any(lora_stats_path(rank, subject).exists() for subject in SUBJECTS)
    ]


def save_figure(fig, name: str):
    """Save every stability figure in PNG and PDF formats."""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def write_context_stability_table():
    """Write the contextual-minus-wordlevel BERT metric table for the report."""
    rows = []
    for subject in SUBJECTS:
        contextual = load_stats("bert_pretrained", subject)
        wordlevel = load_stats("bert_pretrained_wordlevel", subject)
        if contextual is None or wordlevel is None:
            continue
        row = {
            "subject": subject,
            **{f"contextual_{m}": contextual[m] for m in METRICS},
            **{f"wordlevel_{m}": wordlevel[m] for m in METRICS},
            **{f"gain_{m}": contextual[m] - wordlevel[m] for m in METRICS},
        }
        rows.append(row)

    if not rows:
        print("Skipping context stability table: missing contextual or word-level BERT stats.")
        return

    out = METRIC_DIR / "context_stability_bert_wordlevel.csv"
    headers = ["subject"]
    for prefix in ["contextual", "wordlevel", "gain"]:
        headers.extend([f"{prefix}_{m}" for m in METRICS])

    with open(out, "w") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            vals = [row["subject"]]
            vals.extend(f"{row[h]:.6f}" for h in headers[1:])
            fh.write(",".join(vals) + "\n")
    print(f"Saved: {out}")


def fig_context_metric_comparison():
    """Plot aggregate CC metrics for contextual BERT versus low-context BERT."""
    models = available_context_models()
    if len(models) < 2:
        print("Skipping context metric comparison: need contextual and word-level BERT stats.")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(17.2, 5.3), constrained_layout=False)
    x = np.arange(len(SUBJECTS))
    width = 0.34

    for ax, metric in zip(axes, METRICS):
        for offset, model in zip([-width / 2, width / 2], ["bert_pretrained_wordlevel", "bert_pretrained"]):
            vals = []
            for subject in SUBJECTS:
                stats = load_stats(model, subject)
                vals.append(np.nan if stats is None else stats[metric])
            spec = CONTEXT_MODELS[model]
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=spec["color"],
                edgecolor="white",
                linewidth=0.7,
                label=spec["label"],
            )
            for bar, val in zip(bars, vals):
                if np.isfinite(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SUBJECT_LABELS[s] for s in SUBJECTS])
        ax.set_ylabel("Pearson r")
        ax.margins(y=0.22)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.24, wspace=0.24)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.04))
    fig.suptitle("Stability of the Contextual BERT Embedding Choice", fontsize=16, fontweight="bold", y=0.94)
    save_figure(fig, "stability_contextual_vs_wordlevel_metrics")


def fig_context_delta_distribution():
    """Plot voxel-wise CC gains from preserving BERT story context."""
    if not all(corrs_path(model, subject).exists() for model in CONTEXT_MODELS for subject in SUBJECTS):
        print("Skipping context delta distribution: missing contextual or word-level BERT correlations.")
        return

    fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(7.0 * len(SUBJECTS), 5.0), sharey=True, constrained_layout=True)
    if len(SUBJECTS) == 1:
        axes = [axes]

    bins = np.linspace(-0.08, 0.12, 90)
    for ax, subject in zip(axes, SUBJECTS):
        contextual = np.asarray(load_corrs("bert_pretrained", subject), dtype=np.float32)
        wordlevel = np.asarray(load_corrs("bert_pretrained_wordlevel", subject), dtype=np.float32)
        n = min(len(contextual), len(wordlevel))
        delta = contextual[:n] - wordlevel[:n]
        improved = 100.0 * float((delta > 0).mean())
        mean_delta = float(delta.mean())

        ax.hist(delta, bins=bins, density=True, color="#4C86BD", alpha=0.78)
        ax.axvline(0, color="#111111", linewidth=1.0)
        ax.axvline(mean_delta, color="white", linewidth=2.3)
        ax.axvline(mean_delta, color="#111111", linewidth=1.0, linestyle="--")
        ax.set_title(SUBJECT_LABELS[subject], fontsize=13, fontweight="bold")
        ax.set_xlabel("Contextual BERT CC - word-level BERT CC")
        ax.set_ylabel("Density")
        ax.text(
            0.03,
            0.94,
            f"mean gain = {mean_delta:+.4f}\nvoxels improved = {improved:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "#D9D9D9", "boxstyle": "round,pad=0.28"},
        )

    fig.suptitle("Voxel-Wise Contextual Gain over Word-Level BERT", fontsize=16, fontweight="bold")
    save_figure(fig, "stability_contextual_vs_wordlevel_delta")


def fig_context_voxel_distributions():
    """Show how context changes the full voxel-wise CC distribution."""
    models = available_context_models()
    if len(models) < 2:
        print("Skipping context voxel distributions: need contextual and word-level BERT correlations.")
        return

    order = ["bert_pretrained_wordlevel", "bert_pretrained"]
    fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(6.5 * len(SUBJECTS), 5.1), sharey=True, constrained_layout=True)
    if len(SUBJECTS) == 1:
        axes = [axes]

    for ax, subject in zip(axes, SUBJECTS):
        data, labels, colors = [], [], []
        for model in order:
            corrs = load_corrs(model, subject)
            if corrs is None:
                continue
            data.append(np.asarray(corrs, dtype=np.float32))
            labels.append(CONTEXT_MODELS[model]["label"])
            colors.append(CONTEXT_MODELS[model]["color"])

        if not data:
            ax.set_visible(False)
            continue

        box = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False,
            widths=0.58,
            medianprops={"color": "#111111", "linewidth": 1.6},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#2A2A2A")
            patch.set_alpha(0.78)

        for i, corrs in enumerate(data, start=1):
            p95 = np.percentile(corrs, 95)
            p99 = np.percentile(corrs, 99)
            ax.scatter([i], [p95], marker="D", s=34, color="white", edgecolor="#111111", zorder=3)
            ax.scatter([i], [p99], marker="^", s=44, color="#111111", edgecolor="white", zorder=3)
            ax.text(i + 0.08, p99, f"{p99:.3f}", fontsize=8.5, va="center")

        ax.axhline(WELL_PREDICTED_THRESHOLD, color="#D00000", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_title(SUBJECT_LABELS[subject], fontsize=13, fontweight="bold")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=16, ha="right")
        ax.set_ylabel("Voxel-wise held-out CC")
        ax.set_ylim(-0.18, 0.28)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    fig.suptitle("Voxel-Wise CC Distributions for Context Stability", fontsize=16, fontweight="bold")
    save_figure(fig, "stability_contextual_vs_wordlevel_distributions")


def fig_lora_rank_metrics():
    """Plot summary metrics across LoRA ranks when rank-stability outputs exist."""
    ranks = available_lora_ranks()
    if len(ranks) < 2:
        print("Skipping LoRA rank stability metrics: need at least two LoRA ranks.")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(17.2, 4.8), constrained_layout=True)
    x = np.arange(len(ranks))

    for ax, metric in zip(axes, METRICS):
        for subject in SUBJECTS:
            vals = []
            for rank in ranks:
                stats = load_lora_stats(rank, subject)
                vals.append(np.nan if stats is None else float(stats[metric]))
            if np.all(~np.isfinite(vals)):
                continue
            ax.plot(x, vals, marker=SUBJECT_MARKERS[subject], markersize=7, linewidth=2.0, label=SUBJECT_LABELS[subject])
            for xi, val in zip(x, vals):
                if np.isfinite(val):
                    ax.text(xi, val, f" {val:.3f}", va="center", fontsize=8.2)

        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"r={rank}" for rank in ranks])
        ax.set_xlabel("LoRA rank")
        ax.set_ylabel("Pearson r")
        ax.margins(y=0.22)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Stability Check: LoRA Rank Sensitivity", fontsize=16, fontweight="bold")
    save_figure(fig, "stability_lora_rank_metrics")


def fig_lora_rank_distributions():
    """Compare voxel-wise CC distributions across LoRA ranks."""
    ranks = available_lora_ranks()
    if len(ranks) < 2:
        print("Skipping LoRA rank stability distributions: need at least two LoRA ranks.")
        return

    fig, axes = plt.subplots(1, len(SUBJECTS), figsize=(7.0 * len(SUBJECTS), 5.3), sharey=True, constrained_layout=True)
    if len(SUBJECTS) == 1:
        axes = [axes]

    for ax, subject in zip(axes, SUBJECTS):
        data, labels, colors = [], [], []
        for rank in ranks:
            corrs = load_lora_corrs(rank, subject)
            if corrs is None:
                continue
            data.append(np.asarray(corrs, dtype=np.float32))
            labels.append(f"r={rank}")
            colors.append(LORA_RANK_MODELS[rank]["color"])

        if not data:
            ax.set_visible(False)
            continue

        box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.58, medianprops={"color": "#111111", "linewidth": 1.6})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#2A2A2A")
            patch.set_alpha(0.78)

        for i, corrs in enumerate(data, start=1):
            p95 = np.percentile(corrs, 95)
            p99 = np.percentile(corrs, 99)
            ax.scatter([i], [p95], marker="D", s=34, color="white", edgecolor="#111111", zorder=3)
            ax.scatter([i], [p99], marker="^", s=44, color="#111111", edgecolor="white", zorder=3)
            ax.text(i + 0.08, p99, f"{p99:.3f}", fontsize=8.5, va="center")

        ax.axhline(WELL_PREDICTED_THRESHOLD, color="#D00000", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_title(SUBJECT_LABELS[subject], fontsize=13, fontweight="bold")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Voxel-wise held-out CC")
        ax.set_ylim(-0.18, 0.28)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    fig.suptitle("Stability Check: Voxel-Wise CC Across LoRA Ranks", fontsize=16, fontweight="bold")
    save_figure(fig, "stability_lora_rank_distributions")


def main():
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"FIG_DIR:     {FIG_DIR}")

    write_context_stability_table()
    fig_context_metric_comparison()
    fig_context_delta_distribution()
    fig_context_voxel_distributions()
    fig_lora_rank_metrics()
    fig_lora_rank_distributions()

    print("\nDone.")


if __name__ == "__main__":
    main()
