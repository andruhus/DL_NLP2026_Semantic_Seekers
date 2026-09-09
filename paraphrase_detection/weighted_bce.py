import ast

import pandas as pd
import torch
from torch import nn


UNUSED_PARAPHRASE_TYPE_IDS = {12, 19, 20, 23, 27}
VALID_PARAPHRASE_TYPE_IDS = sorted(
    set(range(1, 32)) - UNUSED_PARAPHRASE_TYPE_IDS
)


def create_weighted_bce_loss(pos_weights):
    """Create the weighted binary cross-entropy loss used for training."""
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


def print_label_statistics(positive_counts, negative_counts, pos_weight_sets):
    """Print training-label counts and every configured positive-weight set."""
    statistics = {
        "label_id": VALID_PARAPHRASE_TYPE_IDS,
        "positive": positive_counts.int().tolist(),
        "negative": negative_counts.int().tolist(),
    }
    for name, pos_weights in pos_weight_sets.items():
        statistics[f"{name}_pos_weight"] = pos_weights.tolist()

    print("Training-label statistics:")
    print(
        pd.DataFrame(statistics).to_string(
            index=False, float_format=lambda value: f"{value:.4f}",
        )
    )


def _compute_raw_pos_weights(positive_counts, negative_counts):
    """Compute the inverse-frequency ratio used as the aggressive baseline."""
    verify_positive_training_examples(positive_counts)
    # BCEWithLogitsLoss multiplies the positive loss term, so balancing uses
    # negative / positive (not positive / negative).
    return negative_counts / positive_counts


def compute_aggressive_pos_weights(positive_counts, negative_counts):
    """Compute raw inverse-frequency weights N^- / N^+."""
    return _compute_raw_pos_weights(positive_counts, negative_counts)


def compute_sqrt_pos_weights(positive_counts, negative_counts):
    """Compute square-root inverse-frequency weights."""
    raw_pos_weights = _compute_raw_pos_weights(
        positive_counts, negative_counts,
    )
    return torch.sqrt(raw_pos_weights)


def compute_log_pos_weights(positive_counts, negative_counts):
    """Compute positive logarithmic inverse-frequency weights log(1 + N^- / N^+)."""
    raw_pos_weights = _compute_raw_pos_weights(
        positive_counts, negative_counts,
    )
    return torch.log1p(raw_pos_weights)


def compute_capped_pos_weights(
    positive_counts, negative_counts, max_weight,
):
    """Compute inverse-frequency weights capped at ``max_weight``."""
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    raw_pos_weights = _compute_raw_pos_weights(
        positive_counts, negative_counts,
    )
    return torch.clamp(raw_pos_weights, max=max_weight)


def compute_pos_weights(positive_counts, negative_counts):
    """Backward-compatible alias for aggressive inverse-frequency weights."""
    return compute_aggressive_pos_weights(positive_counts, negative_counts)


def verify_positive_training_examples(positive_counts):
    """Fail early if a label has no positive example in the training split."""
    missing_labels = [
        type_id
        for type_id, count in zip(VALID_PARAPHRASE_TYPE_IDS, positive_counts)
        if count.item() == 0
    ]
    if missing_labels:
        raise ValueError(
            "Every label must have a positive training example. "
            f"Missing paraphrase type IDs: {missing_labels}"
        )


def encode_paraphrase_labels(dataset):
    """Convert paraphrase type IDs into one multi-hot vector per example."""
    labels = []
    for type_ids in dataset["paraphrase_type_ids"]:
        type_set = set(ast.literal_eval(str(type_ids)))
        labels.append(
            [int(type_id in type_set) for type_id in VALID_PARAPHRASE_TYPE_IDS]
        )
    return torch.tensor(labels, dtype=torch.float)


def count_label_examples(labels):
    """Count positive and negative training examples for each label."""
    if labels.ndim != 2 or labels.shape[1] != len(VALID_PARAPHRASE_TYPE_IDS):
        raise ValueError(
            "Expected labels with shape "
            f"(num_examples, {len(VALID_PARAPHRASE_TYPE_IDS)}), "
            f"but got {tuple(labels.shape)}."
        )
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must contain only binary values (0 or 1).")

    positive_counts = labels.sum(dim=0)
    negative_counts = labels.shape[0] - positive_counts
    return positive_counts, negative_counts
