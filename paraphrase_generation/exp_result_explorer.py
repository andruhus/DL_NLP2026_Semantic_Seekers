from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 200)

RESULTS_PATH = Path(__file__).with_name("run_5epoch_results.csv")
TRAINING_RESULTS_PATH = Path(__file__).with_name("train_5epoch_results.csv")


def _validate_ids(ids, training_results):
    if isinstance(ids, (int, str)):
        ids = [ids]
    elif not isinstance(ids, Iterable):
        raise TypeError("ids must be an experiment ID or an iterable of IDs")

    ids = [int(experiment_id) for experiment_id in ids]
    if not ids:
        raise ValueError("ids must contain at least one experiment ID")

    available_ids = set(training_results["id"])
    missing_ids = sorted(set(ids) - available_ids)
    if missing_ids:
        raise ValueError(f"No training data found for experiment IDs: {missing_ids}")
    return ids


def plot_training_metrics(
    ids: int | str | Iterable[int | str],
    show: bool = True,
    baseline: bool = False,
):
    """Plot learning rate and development BLEU metrics for one or more IDs.

    Args:
        ids: One experiment ID or an iterable of experiment IDs from
            ``run_5epoch_results.csv``.
        show: Whether to display the Matplotlib figure immediately.
        baseline: Include the constant 2e-5 run, found in the summary CSV,
            as a dashed baseline. It is not duplicated if already requested.

    Returns:
        A tuple containing the Matplotlib figure and its 2-by-3 axes array.
    """
    training_results = pd.read_csv(TRAINING_RESULTS_PATH)
    summary_results = pd.read_csv(RESULTS_PATH).set_index("id")
    ids = list(dict.fromkeys(_validate_ids(ids, training_results)))
    baseline_id = None
    if baseline:
        candidates = summary_results.index[
            (summary_results["scheduler_type"] == "Constant learning rate")
            & (summary_results["lr"] == 2e-5)
        ].tolist()
        if len(candidates) != 1:
            raise ValueError("Expected exactly one constant 2e-5 baseline in the summary CSV.")
        baseline_id = int(candidates[0])
        if baseline_id not in ids:
            ids.append(baseline_id)
        _validate_ids(ids, training_results)

    figure, axes = plt.subplots(3, 2, figsize=(18, 8), sharex=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    id_colors = {
        experiment_id: color_cycle[index % len(color_cycle)]
        for index, experiment_id in enumerate(ids)
    }
    if baseline_id is not None:
        id_colors[baseline_id] = "black"
    plots = (
        (axes[0, 0], "next_lr", "Next learning rate"),
        (axes[2, 0], "dev_reference_bleu", "Reference BLEU"),
        (axes[2, 1], "dev_input_bleu", "Input BLEU"),
        (axes[1, 0], "dev_penalized_bleu", "Penalized BLEU"),
        (axes[0, 1], "loss", "Training loss"),
    )

    for index, experiment_id in enumerate(ids):
        experiment_rows = training_results[
            training_results["id"] == experiment_id
        ].sort_values(by=["epoch"])
        is_baseline = experiment_id == baseline_id
        label = f"Baseline (ID {experiment_id})" if is_baseline else f"ID {experiment_id}"
        for axis, column, title in plots:
            axis.plot(
                experiment_rows["epoch"],
                experiment_rows[column],
                marker="x" if is_baseline else ("o", "s", "^")[index % 3],
                markerfacecolor="none",
                linestyle="--" if is_baseline else ("-", ":", "-.")[index % 3],
                color=id_colors[experiment_id],
                label=label,
            )
            axis.set_title(title)
            axis.set_ylabel(column)
            axis.grid(True, alpha=0.3)

    parameter_columns = (
        "scheduler_type",
        "lr",
        "min_lr",
        "decay_epochs",
        "step_size",
        "gamma",
        "total_steps",
        "inverse_warm_up_step",
        "patience",
        "threshold",
    )
    parameter_sections = []
    for experiment_id in ids:
        metadata = summary_results.loc[experiment_id]
        scheduler_type = str(metadata["scheduler_type"])
        parameter_values = []
        for column in parameter_columns[1:]:
            value = metadata[column]
            if pd.notna(value):
                display_column = (
                    "factor"
                    if column == "gamma" and scheduler_type == "Metric dependent"
                    else column
                )
                parameter_values.append(f"{display_column}={value:g}")
        compact_lines = [
            "; ".join(parameter_values[index:index + 2])
            for index in range(0, len(parameter_values), 2)
        ]
        parameter_sections.append(
            (experiment_id, [scheduler_type, *compact_lines])
        )

    parameter_axis = axes[1, 1]
    parameter_axis.set_title("Experiment parameters")
    parameter_axis.axis("off")
    section_height = 1.0 / len(parameter_sections)
    for index, (experiment_id, parameter_lines) in enumerate(parameter_sections):
        text_y = 1.0 - index * section_height
        parameter_axis.text(
            0.0,
            text_y,
            f"Baseline (ID {experiment_id})" if experiment_id == baseline_id else f"ID {experiment_id}",
            color=id_colors[experiment_id],
            transform=parameter_axis.transAxes,
            verticalalignment="top",
            fontsize=8,
            fontweight="bold",
        )
        parameter_axis.text(
            0.0,
            text_y - 0.06,
            "\n".join(parameter_lines),
            transform=parameter_axis.transAxes,
            verticalalignment="top",
            fontsize=7.5,
            linespacing=1.0,
        )

    for axis, _, _ in plots:
        axis.set_xlabel("Epoch")
        axis.set_xticks(sorted(training_results.loc[training_results["id"].isin(ids), "epoch"].unique()))
        axis.legend()

    figure.suptitle("Training metrics by experiment ID")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes

def _format_scientific(value):
    if pd.isna(value):
        return ""
    return format(Decimal(str(value)).normalize(), "e")

def scatter_plot_losses(show=True):
    df = pd.read_csv(TRAINING_RESULTS_PATH)
    ref_lim = 46.5
    pen_lim = 10
    df = df[df["dev_reference_bleu"] >= ref_lim]
    df = df[df["dev_penalized_bleu"] >= pen_lim]
    df = df[df["loss"] <= 2.5]
    figure, (ref_axis, pen_axis) = plt.subplots(
        1, 2,
        figsize=(14, 5),
    )

    epochs = np.arange(1, 6)
    cmap = plt.get_cmap("viridis", len(epochs))
    norm = BoundaryNorm(
        boundaries=np.arange(0.5, 5.6, 1.0),
        ncolors=len(epochs),
    )

    ref_scatter = ref_axis.scatter(
        df["loss"],
        df["dev_reference_bleu"],
        c=df["epoch"],
        cmap=cmap,
        norm=norm,
        s=70,
    )

    pen_axis.scatter(
        df["loss"],
        df["dev_penalized_bleu"],
        c=df["epoch"],
        cmap=cmap,
        norm=norm,
        s=70,
    )

    centroids = df.groupby("epoch")[[
        "loss",
        "dev_reference_bleu",
        "dev_penalized_bleu",
    ]].mean()
    centroid_epochs = centroids.index.to_numpy()
    ref_axis.scatter(
        centroids["loss"],
        centroids["dev_reference_bleu"],
        c=centroid_epochs,
        cmap=cmap,
        norm=norm,
        s=180,
        marker="x",
        zorder=3,
        label="Epoch centroid",
    )
    pen_axis.scatter(
        centroids["loss"],
        centroids["dev_penalized_bleu"],
        c=centroid_epochs,
        cmap=cmap,
        norm=norm,
        s=180,
        marker="x",
        zorder=3,
        label="Epoch centroid",
    )

    ref_axis.set(
        title="Reference BLEU vs. Training Loss",
        xlabel="Training loss",
        ylabel="Reference BLEU",
        xlim=(0, 2.5),
        ylim=(ref_lim-1, 49),
    )

    pen_axis.set(
        title="Penalized BLEU vs. Training Loss",
        xlabel="Training loss",
        ylabel="Penalized BLEU",
        xlim=(0, 2.5),
        ylim=(pen_lim -1, 25),
    )

    for axis in (ref_axis, pen_axis):
        axis.set_xticks(np.arange(0, 2.6, 0.5))
        axis.grid(alpha=0.3)
        axis.legend()

    figure.colorbar(
        ref_scatter,
        ax=(ref_axis, pen_axis),
        ticks=epochs,
        label="Epoch",
    )
    figure.tight_layout()

    if show:
        plt.show()

    return figure, (ref_axis, pen_axis)



def print_info(sort_col):
    results = pd.read_csv(RESULTS_PATH)
    sorted = results.sort_values(
        sort_col,
        ascending=False,
    )[[
        "id",
        "scheduler_type",
        "lr",
        "min_lr",
        "dev_reference_bleu",
        "dev_input_bleu",
        "dev_penalized_bleu",
    ]]
    display_results = sorted.copy()
    display_results["lr"] = [
        _format_scientific(value) for value in display_results["lr"]
    ]
    display_results["min_lr"] = [
        _format_scientific(value) for value in display_results["min_lr"]
    ]
    print(display_results.to_string(index=False))

if __name__ == "__main__":
    # col = "dev_penalized_bleu"
    # col = "dev_reference_bleu"
    # print_info(col)
    # plot_training_metrics([25,70,65])
    scatter_plot_losses()
