from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
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
):
    """Plot learning rate and development BLEU metrics for one or more IDs.

    Args:
        ids: One experiment ID or an iterable of experiment IDs from
            ``run_5epoch_results.csv``.
        show: Whether to display the Matplotlib figure immediately.

    Returns:
        A tuple containing the Matplotlib figure and its 2-by-3 axes array.
    """
    training_results = pd.read_csv(TRAINING_RESULTS_PATH)
    summary_results = pd.read_csv(RESULTS_PATH).set_index("id")
    ids = _validate_ids(ids, training_results)
    missing_metadata_ids = sorted(set(ids) - set(summary_results.index))
    if missing_metadata_ids:
        raise ValueError(
            f"No experiment metadata found for IDs: {missing_metadata_ids}"
        )

    figure, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    id_colors = {
        experiment_id: color_cycle[index % len(color_cycle)]
        for index, experiment_id in enumerate(ids)
    }
    plots = (
        (axes[0, 0], "next_lr", "Next learning rate"),
        (axes[0, 1], "dev_reference_bleu", "Reference BLEU"),
        (axes[0, 2], "dev_input_bleu", "Input BLEU"),
        (axes[1, 0], "dev_penalized_bleu", "Penalized BLEU"),
        (axes[1, 2], "loss", "Training loss"),
    )

    for experiment_id in ids:
        experiment_rows = training_results[
            training_results["id"] == experiment_id
        ].sort_values("epoch")
        label = f"ID {experiment_id}"
        for axis, column, title in plots:
            axis.plot(
                experiment_rows["epoch"],
                experiment_rows[column],
                marker="o",
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
    )
    parameter_sections = []
    for experiment_id in ids:
        metadata = summary_results.loc[experiment_id]
        parameter_lines = []
        for column in parameter_columns:
            value = metadata[column]
            if pd.notna(value):
                formatted_value = (
                    str(value) if column == "scheduler_type" else f"{value:g}"
                )
                parameter_lines.append(f"{column}: {formatted_value}")
        parameter_sections.append((experiment_id, parameter_lines))

    parameter_axis = axes[1, 1]
    parameter_axis.set_title("Experiment parameters")
    parameter_axis.axis("off")
    line_height = 0.05
    text_y = 1.0
    for experiment_id, parameter_lines in parameter_sections:
        parameter_axis.text(
            0.0,
            text_y,
            f"ID {experiment_id}",
            color=id_colors[experiment_id],
            transform=parameter_axis.transAxes,
            verticalalignment="top",
            fontsize=9,
        )
        text_y -= line_height
        parameter_axis.text(
            0.0,
            text_y,
            "\n".join(parameter_lines),
            transform=parameter_axis.transAxes,
            verticalalignment="top",
            fontsize=9,
        )
        text_y -= line_height * (len(parameter_lines) + 1)

    for axis, _, _ in plots:
        axis.set_xlabel("Epoch")
        axis.legend()

    figure.suptitle("Training metrics by experiment ID")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axes

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
    print(sorted.to_string(index=False))

if __name__ == "__main__":
    # col = "dev_penalized_bleu"
    col = "dev_reference_bleu"
    print_info(col)
    # plot_training_metrics([30,25,16])
