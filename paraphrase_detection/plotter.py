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
    y_focal: dict[float, list[float]] | None = None,
):
    epochs = list(range(1, 26))
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
                f"Focal Loss (gamma = {gamma:g})",
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

if __name__ == '__main__':
    exp1()