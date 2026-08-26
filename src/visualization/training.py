"""Small plotting helpers for the RevIN experiment."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history, criterion, path, plot_step_train_loss=False):
    fig, ax = plt.subplots(figsize=(9, 4))
    if plot_step_train_loss and history.get("train"):
        ax.plot(range(1, len(history["train"]) + 1), history["train"], label="train step")
    interval = history.get("train_batch") or []
    if interval:
        ax.plot(
            [item["step"] for item in interval],
            [item["losses"][criterion] for item in interval],
            marker="o",
            label="train interval average",
        )
    for split, values in (history.get("valid") or {}).items():
        if values:
            ax.plot(
                [item["step"] for item in values],
                [item["losses"][criterion] for item in values],
                marker="o",
                label=split,
            )
    ax.set(xlabel="optimizer step", ylabel=criterion, yscale="log")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
