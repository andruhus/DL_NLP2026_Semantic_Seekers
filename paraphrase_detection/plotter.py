from pathlib import Path

import matplotlib.pyplot as plt

# Output directory
FIGURE_DIR = Path("paraphrase_detection/figure")
FIGURE_DIR.mkdir(exist_ok=True)

epochs = list(range(1, 26))

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


def save_plot(y_unweighted, y_weighted, ylabel, filename):
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        y_unweighted,
        marker="o",
        label="Unweighted BCE",
    )
    plt.plot(
        epochs,
        y_weighted,
        marker="o",
        label="Weighted BCE",
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


save_plot(
    unweighted_dev_acc,
    weighted_dev_acc,
    "Dev Accuracy",
    "dev_acc.png",
)

save_plot(
    unweighted_mcc,
    weighted_mcc,
    "MCC",
    "mcc.png",
)
