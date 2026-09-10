"""Generate learning-rate scheduler figures for the experiment report."""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .exp_result_explorer import plot_training_metrics, scatter_plot_losses
else:
    from exp_result_explorer import plot_training_metrics, scatter_plot_losses


INITIAL_LR = 2e-5
MIN_LR = 0.0
EPOCHS = 5
TRAIN_EXAMPLES = 2_457
BATCH_SIZE = 8
STEPS_PER_EPOCH = math.ceil(TRAIN_EXAMPLES / BATCH_SIZE)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
STEP_DECAY_EPOCHS = 1
STEP_GAMMA = 0.5
WARMUP_STEPS = 100
METRIC_FACTOR = 0.5
METRIC_PATIENCE = 1

FIGURE_DIR = Path(__file__).resolve().parent.parent / "figure" / "ptg"
RESULTS_PATH = Path(__file__).resolve().parent / "run_5epoch_results.csv"
SCHEDULER_COMPARISONS = (
    ("Constant learning rate", "Constant learning rate", "constant_training_comparison.png"),
    ("Step decay", "Step decay", "step_training_comparison.png"),
    ("Cosine decay", "Cosine decay", "cosine_training_comparison.png"),
    ("Linear decay", "Linear decay", "linear_training_comparison.png"),
    ("Inverse square root", "Inverse-square-root decay", "inverse_sqrt_training_comparison.png"),
    ("Metric dependent", "Metric-dependent decay", "metric_training_comparison.png"),
)

# Choose the experiment IDs to compare for each scheduler. These start with the
# current top two by penalized BLEU, but can contain any number of valid IDs.
SELECTED_COMPARISON_IDS = {
    "Constant learning rate": (74, 65),
    "Step decay": (16, 17),
    "Cosine decay": (46, 51),
    "Linear decay": (38, 33),
    "Inverse square root": (54, 63),
    "Metric dependent": (91, 97),
}


def save_update_plot(rates, title, filename, *, mark_warmup=False):
    """Plot an update-based schedule against equivalent training epochs."""
    updates = np.arange(len(rates))
    epochs = updates / STEPS_PER_EPOCH

    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    ax.plot(epochs, rates, linewidth=2.4)
    if mark_warmup:
        warmup_epoch = WARMUP_STEPS / STEPS_PER_EPOCH
        ax.axvline(
            warmup_epoch,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label=f"Warmup ends (update {WARMUP_STEPS})",
        )
        ax.legend()
    ax.set(
        title=title,
        xlabel="Training epoch",
        ylabel="Learning rate",
        xlim=(0, EPOCHS),
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_update_based_schedules():
    """Generate figures for the five schedules updated after optimizer steps."""
    updates = np.arange(TOTAL_STEPS + 1)
    progress = np.minimum(updates, TOTAL_STEPS) / TOTAL_STEPS

    constant = np.full_like(updates, INITIAL_LR, dtype=float)
    save_update_plot(
        constant,
        "Constant learning rate",
        "constant_learning_rate.png",
    )

    step_size = STEPS_PER_EPOCH * STEP_DECAY_EPOCHS
    step = np.maximum(
        MIN_LR,
        INITIAL_LR * STEP_GAMMA ** np.floor_divide(updates, step_size),
    )
    save_update_plot(step, "Step decay", "step_decay.png")

    cosine = MIN_LR + 0.5 * (INITIAL_LR - MIN_LR) * (
        1.0 + np.cos(np.pi * progress)
    )
    save_update_plot(cosine, "Cosine decay", "cosine_decay.png")

    linear = MIN_LR + (INITIAL_LR - MIN_LR) * (1.0 - progress)
    save_update_plot(linear, "Linear decay", "linear_decay.png")

    update_number = updates + 1
    inverse_sqrt_scale = np.minimum(
        update_number / WARMUP_STEPS,
        np.sqrt(WARMUP_STEPS / update_number),
    )
    inverse_sqrt = np.maximum(MIN_LR, INITIAL_LR * inverse_sqrt_scale)
    save_update_plot(
        inverse_sqrt,
        "Inverse-square-root decay with linear warmup",
        "inverse_square_root.png",
        mark_warmup=True,
    )


def plot_metric_dependent_schedule():
    """Plot the default metric scheduler for an illustrative metric trajectory."""
    # Improvement in epochs 1 and 2, followed by two non-improving evaluations.
    illustrative_metrics = [10.0, 12.0, 12.0, 12.0, 13.0]
    learning_rates = []
    current_lr = INITIAL_LR
    best_metric = None
    bad_epochs = 0

    for metric in illustrative_metrics:
        learning_rates.append(current_lr)
        if best_metric is None or metric > best_metric + 1e-8:
            best_metric = metric
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > METRIC_PATIENCE:
                current_lr = max(MIN_LR, current_lr * METRIC_FACTOR)
                bad_epochs = 0

    epochs = np.arange(1, EPOCHS + 1)
    fig, ax = plt.subplots(figsize=(5.4, 3.15))
    ax.step(epochs, learning_rates, where="mid", linewidth=2.4)
    ax.scatter(epochs, learning_rates, s=34, zorder=3)
    ax.annotate(
        "Two consecutive\nnon-improving evaluations",
        xy=(5, learning_rates[-1]),
        xytext=(3.2, 1.35e-5),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2},
    )
    ax.set(
        title="Metric-dependent decay (illustrative trajectory)",
        xlabel="Training epoch",
        ylabel="Learning rate used during epoch",
        xlim=(0.75, EPOCHS + 0.25),
        xticks=epochs,
        ylim=(0.0, INITIAL_LR * 1.1),
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "metric_dependent.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_scheduler_comparisons():
    """Compare manually selected scheduler runs with the constant 2e-5 baseline."""
    with RESULTS_PATH.open(newline="", encoding="utf-8") as source:
        results = list(csv.DictReader(source))

    selected_ids = {}
    for scheduler_type, title, filename in SCHEDULER_COMPARISONS:
        ids = list(SELECTED_COMPARISON_IDS.get(scheduler_type, ()))


        selected_ids[scheduler_type] = ids

        figure, _ = plot_training_metrics(ids, show=False, baseline=True)
        figure.suptitle(f"{title}: selected runs vs. constant 2e-5 baseline")
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(FIGURE_DIR / filename, dpi=600, bbox_inches="tight")
        plt.close(figure)
    return selected_ids



def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_update_based_schedules()
    plot_metric_dependent_schedule()
    selected_ids = plot_scheduler_comparisons()
    for scheduler_type, ids in selected_ids.items():
        print(f"{scheduler_type}: selected IDs {', '.join(map(str, ids))}")

    loss_bleu_figure, _ = scatter_plot_losses(show=False)
    loss_bleu_figure.savefig(
        FIGURE_DIR / "loss_bleu_scatter.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(loss_bleu_figure)
    print(f"Saved loss/BLEU scatter plot to {FIGURE_DIR / 'loss_bleu_scatter.png'}")
    print(f"Saved scheduler plots to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
