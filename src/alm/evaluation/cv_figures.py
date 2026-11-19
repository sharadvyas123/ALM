"""Reproducible diagnostic figures for grouped cross-validation runs."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from ..data.provenance import CLASS_NAMES


def fold_curves(history: list[dict], output: Path, fold_name: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(epochs, [r["train_total_loss"] for r in history], label="Training")
    axes[0].plot(epochs, [r["val_total_loss"] for r in history], label="Inner validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_mAP"] for r in history], color="#2368a2", label="Inner validation mAP")
    axes[1].set(title="Checkpoint selection metric", xlabel="Epoch", ylabel="mAP", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle(f"{fold_name}: training history")
    fig.savefig(output, dpi=170)
    plt.close(fig)


def outer_pr_curve(targets: np.ndarray, scores: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for index, name in enumerate(CLASS_NAMES):
        if targets[:, index].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(targets[:, index], scores[:, index])
        ax.plot(recall, precision, label=name, linewidth=1.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
           title="Outer-fold precision-recall curves (out-of-fold predictions)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def fold_comparison(metrics: dict[str, dict], output: Path) -> None:
    names = list(metrics)
    keys = ("mAP", "macro_f1", "micro_f1")
    x = np.arange(len(names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, key in enumerate(keys):
        values = [metrics[name][key] for name in names]
        ax.bar(x + (i - 1) * width, values, width, label=key)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Outer-fold performance by fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def class_metric_heatmap(metrics: dict[str, dict], output: Path) -> None:
    folds = list(metrics)
    values = np.asarray([[metrics[f]["per_class"][name]["f1"] for name in CLASS_NAMES]
                         for f in folds], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(folds)), folds)
    ax.set_title("Outer-fold per-class F1 (thresholds selected on inner validation)")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="F1")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def spectrogram_examples(dataset, output: Path, fold_name: str) -> None:
    """Render one deterministic clean, two-source, and three-source example."""
    chosen = []
    for kind in ("clean", "2-class", "3-class"):
        index = next((i for i, row in enumerate(dataset.rows) if row["mix_type"] == kind), None)
        if index is not None:
            chosen.append((kind, index))
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 3.2 * len(chosen)), squeeze=False,
                             constrained_layout=True)
    for row_index, (kind, sample_index) in enumerate(chosen):
        item = dataset[sample_index]
        mel = item["mel"][0].numpy()
        axes[row_index, 0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        axes[row_index, 0].set(title=f"{kind}: log-mel", ylabel="Mel bin", xlabel="Frame")
        if "mask_target" in item:
            mask = item["mask_target"].sum(0).numpy()
            axes[row_index, 1].imshow(mask, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
            axes[row_index, 1].set(title="Summed source contribution mask", ylabel="Mel bin", xlabel="Frame")
        else:
            axes[row_index, 1].axis("off")
    fig.suptitle(f"{fold_name}: representative outer-fold samples")
    fig.savefig(output, dpi=170)
    plt.close(fig)
