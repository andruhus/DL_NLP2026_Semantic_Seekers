from pathlib import Path

import matplotlib.pyplot as plt

# Output directory
FIGURE_DIR = Path("paraphrase_detection/figure")
FIGURE_DIR.mkdir(exist_ok=True)

def __plot_graph(
    y_unweighted,
    ylabel,
    filename,
    y_weighted=None,
    y_weighted_sqrt=None,
    y_weighted_log=None,
    y_weighted_capped=None,
    y_focal: dict[float, list[float] | None] | None = None,
):
    epochs = list(range(1, len(y_unweighted) + 1))
    plt.figure(figsize=(8, 5))

    curves = [
        ("Unweighted BCE", y_unweighted),
        ("Weighted BCE", y_weighted),
        ("Square-Root Weighted BCE", y_weighted_sqrt),
        ("Logarithmic Weighted BCE", y_weighted_log),
        ("Capped Weighted BCE", y_weighted_capped),
    ]
    if y_focal is not None:
        curves.extend(
            (
                rf"Focal Loss ($\gamma = {gamma:g}$)",
                values,
            )
            for gamma, values in y_focal.items()
        )

    for label, values in curves:
        if values is not None:
            plt.plot(epochs, values, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def __plot_focal_best_performance(
    focal_accuracy,
    focal_mcc,
    filename,
):
    gammas = list(focal_accuracy)
    positions = list(range(len(gammas)))
    width = 0.4

    plt.figure(figsize=(11, 5))
    plt.bar(
        [position - width / 2 for position in positions],
        [focal_accuracy[gamma] for gamma in gammas],
        width=width,
        label="Development Accuracy",
    )
    plt.bar(
        [position + width / 2 for position in positions],
        [focal_mcc[gamma] for gamma in gammas],
        width=width,
        label="Development MCC",
    )
    plt.xlabel(r"Focal-loss $\gamma$")
    plt.ylabel("Best development-set score")
    plt.xticks(positions, [f"{gamma:g}" for gamma in gammas])
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def exp1():
    unweighted_dev_acc = [
        0.910, 0.915, 0.926, 0.942, 0.956,
        0.964, 0.980, 0.987, 0.989, 0.994,
        0.997, 0.997, 0.999, 0.999, 0.999,
        0.999, 0.999, 1.000, 0.999, 1.000,
        1.000, 1.000, 1.000, 1.000, 0.999,
    ]
    
    weighted_dev_acc = [
        0.556, 0.568, 0.622, 0.594, 0.708,
        0.779, 0.793, 0.838, 0.855, 0.879,
        0.899, 0.910, 0.927, 0.937, 0.947,
        0.951, 0.958, 0.961, 0.966, 0.975,
        0.973, 0.974, 0.979, 0.976, 0.974,
    ]
    
    unweighted_mcc = [
        0.037, 0.092, 0.221, 0.354, 0.461,
        0.594, 0.679, 0.732, 0.744, 0.832,
        0.848, 0.899, 0.947, 0.952, 0.959,
        0.958, 0.959, 0.959, 0.956, 0.960,
        0.960, 0.962, 0.961, 0.962, 0.959,
    ]
    
    weighted_mcc = [
        0.046, 0.110, 0.125, 0.160, 0.219,
        0.330, 0.388, 0.454, 0.534, 0.584,
        0.638, 0.673, 0.738, 0.753, 0.790,
        0.811, 0.847, 0.848, 0.875, 0.890,
        0.881, 0.895, 0.896, 0.901, 0.893,
    ]
    
    __plot_graph(
        unweighted_dev_acc,
        "Dev Accuracy",
        "dev_acc.png",
        y_weighted=weighted_dev_acc,
    )
    
    __plot_graph(
        unweighted_mcc,
        "MCC",
        "mcc.png",
        y_weighted=weighted_mcc,
    )


def exp2():
    unweighted_dev_acc = [
        0.910, 0.915, 0.926, 0.942, 0.956,
        0.964, 0.980, 0.987, 0.989, 0.994,
        0.997, 0.997, 0.999, 0.999, 0.999,
        0.999, 0.999, 1.000, 0.999, 1.000,
        1.000, 1.000, 1.000, 1.000, 0.999,
    ]

    unweighted_mcc = [
        0.037, 0.092, 0.221, 0.354, 0.461,
        0.594, 0.679, 0.732, 0.744, 0.832,
        0.848, 0.899, 0.947, 0.952, 0.959,
        0.958, 0.959, 0.959, 0.956, 0.960,
        0.960, 0.962, 0.961, 0.962, 0.959,
    ]

    soft_weighted_sqrt_dev_acc = [
        0.907, 0.906, 0.906, 0.910, 0.944,
        0.956, 0.969, 0.975, 0.982, 0.987,
        0.991, 0.990, 0.990, 0.991, 0.993,
        0.994, 0.993, 0.997, 0.996, 0.998,
        0.996, 0.997, 0.997, 0.999, 0.997,
    ]
    soft_weighted_sqrt_mcc = [
        0.043, 0.122, 0.381, 0.517, 0.685,
        0.730, 0.801, 0.832, 0.870, 0.906,
        0.921, 0.924, 0.924, 0.929, 0.936,
        0.940, 0.938, 0.949, 0.940, 0.954,
        0.948, 0.950, 0.947, 0.956, 0.945,
    ]
    soft_weighted_log_dev_acc = [
        0.873, 0.861, 0.880, 0.906, 0.935,
        0.943, 0.961, 0.967, 0.972, 0.977,
        0.980, 0.971, 0.981, 0.983, 0.982,
        0.986, 0.986, 0.988, 0.984, 0.991,
        0.989, 0.992, 0.993, 0.994, 0.995,
    ]
    soft_weighted_log_mcc = [
        0.047, 0.126, 0.366, 0.615, 0.745,
        0.769, 0.832, 0.857, 0.873, 0.880,
        0.892, 0.891, 0.902, 0.914, 0.907,
        0.914, 0.917, 0.921, 0.914, 0.923,
        0.920, 0.929, 0.931, 0.933, 0.933,
    ]
    soft_weighted_capped_dev_acc = [
        0.654, 0.771, 0.718, 0.795, 0.873,
        0.886, 0.907, 0.924, 0.938, 0.963,
        0.965, 0.958, 0.965, 0.967, 0.974,
        0.976, 0.972, 0.975, 0.972, 0.980,
        0.980, 0.982, 0.987, 0.987, 0.984,
    ]
    soft_weighted_capped_mcc = [
        0.100, 0.205, 0.311, 0.399, 0.577,
        0.640, 0.686, 0.726, 0.796, 0.825,
        0.850, 0.851, 0.874, 0.874, 0.891,
        0.896, 0.890, 0.898, 0.900, 0.908,
        0.909, 0.908, 0.920, 0.919, 0.914,
    ]

    __plot_graph(
        unweighted_dev_acc,
        "Dev Accuracy",
        "dev_acc_soft_weighted.png",
        y_weighted_sqrt=soft_weighted_sqrt_dev_acc,
        y_weighted_log=soft_weighted_log_dev_acc,
        y_weighted_capped=soft_weighted_capped_dev_acc,
    )

    __plot_graph(
        unweighted_mcc,
        "MCC",
        "mcc_soft_weighted.png",
        y_weighted_sqrt=soft_weighted_sqrt_mcc,
        y_weighted_log=soft_weighted_log_mcc,
        y_weighted_capped=soft_weighted_capped_mcc,
    )


def exp3():
    # The focal-loss experiments ran for five epochs, so use the matching
    # five-epoch unweighted BCE reference curves.
    unweighted_dev_acc = [0.910, 0.915, 0.926, 0.942, 0.956]
    unweighted_mcc = [0.037, 0.092, 0.221, 0.354, 0.461]

    # TODO: Replace each None with its five-epoch metric values for that gamma.
    focal_dev_acc: dict[float, list[float] | None] = {
        0.25: None,
        0.5: None,
        0.62: None,
        0.75: None,
        0.87: None,
        1.0: None,
        1.25: None,
        1.5: None,
        2.0: None,
        4.0: None,
    }
    focal_mcc: dict[float, list[float] | None] = {
        0.25: None,
        0.5: None,
        0.62: None,
        0.75: None,
        0.87: None,
        1.0: None,
        1.25: None,
        1.5: None,
        2.0: None,
        4.0: None,
    }

    __plot_graph(
        unweighted_dev_acc,
        "Dev Accuracy",
        "dev_acc_focal.png",
        y_focal=focal_dev_acc,
    )
    __plot_graph(
        unweighted_mcc,
        "MCC",
        "mcc_focal.png",
        y_focal=focal_mcc,
    )

    focal_best_accuracy = {
        0.25: 0.9505,
        0.5: 0.9507,
        0.62: 0.9391,
        0.75: 0.9107,
        0.87: 0.9563,
        1.0: 0.9567,
        1.25: 0.9577,
        1.5: 0.9542,
        2.0: 0.9462,
        4.0: 0.9232,
    }
    focal_best_mcc = {
        0.25: 0.4201,
        0.5: 0.4372,
        0.62: 0.3571,
        0.75: 0.0380,
        0.87: 0.5085,
        1.0: 0.4606,
        1.25: 0.4992,
        1.5: 0.4486,
        2.0: 0.4130,
        4.0: 0.1830,
    }
    __plot_focal_best_performance(
        focal_best_accuracy,
        focal_best_mcc,
        "focal_best_performance.png",
    )


if __name__ == '__main__':
    exp1()
    exp2()
    exp3()
