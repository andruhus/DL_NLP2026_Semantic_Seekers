#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


RUNS = [
    {
        "label": "Pair features + linear head",
        "job_id": "15552057",
        "epochs": [1, 2, 3],
        "train_loss": [0.461, 0.344, 0.256],
        "train_acc": [0.858, 0.917, 0.954],
        "dev_acc": [0.823, 0.845, 0.850],
    },
    {
        "label": "Pair features + MLP, dropout 0.3",
        "job_id": "15580223",
        "epochs": [1, 2, 3],
        "train_loss": [0.467, 0.346, 0.260],
        "train_acc": [0.845, 0.911, 0.949],
        "dev_acc": [0.813, 0.840, 0.849],
    },
    {
        "label": "Pair features + MLP, dropout 0.1",
        "job_id": "15591323",
        "epochs": [1, 2, 3, 4, 5, 6, 7],
        "train_loss": [0.440, 0.305, 0.218, 0.153, 0.110, 0.083, 0.066],
        "train_acc": [0.864, 0.927, 0.954, 0.972, 0.986, 0.991, 0.993],
        "dev_acc": [0.830, 0.857, 0.858, 0.858, 0.865, 0.868, 0.866],
    },
]


def save_accuracy_plot(output_dir: Path) -> None:
    plt.figure(figsize=(11, 6))

    for run in RUNS:
        label = f"{run['label']} ({run['job_id']})"
        train_line, = plt.plot(
            run["epochs"],
            run["train_acc"],
            marker="o",
            linestyle="--",
            label=f"{label} train",
        )
        plt.plot(
            run["epochs"],
            run["dev_acc"],
            marker="o",
            linestyle="-",
            color=train_line.get_color(),
            label=f"{label} dev",
        )

    plt.title("QQP train and development accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, 8))
    plt.ylim(0.80, 1.00)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "qqp_train_dev_accuracy_curves.png", dpi=200)
    plt.close()


def save_loss_plot(output_dir: Path) -> None:
    plt.figure(figsize=(11, 6))

    for run in RUNS:
        label = f"{run['label']} ({run['job_id']})"
        plt.plot(
            run["epochs"],
            run["train_loss"],
            marker="o",
            label=label,
        )

    plt.title("QQP training loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.xticks(range(1, 8))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "qqp_train_loss_curves.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("figure/qqp"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_accuracy_plot(args.output_dir)
    save_loss_plot(args.output_dir)

    print(f"Wrote figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
