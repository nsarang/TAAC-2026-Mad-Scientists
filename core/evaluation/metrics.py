"""Classification metrics for CVR evaluation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Apply numerically stable sigmoid to `logits`."""
    clipped = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC, returning 0.5 when a class is missing."""
    from sklearn.metrics import roc_auc_score

    binary_labels = (labels > 0.5).astype(np.float64)
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        return 0.5
    return float(roc_auc_score(binary_labels, scores.astype(np.float64)))


def binary_logloss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute mean binary cross-entropy loss."""
    if labels.size == 0 or probabilities.size == 0:
        return 0.0
    clipped = np.clip(probabilities.astype(np.float64), 1.0e-7, 1.0 - 1.0e-7)
    labels = labels.astype(np.float64)
    losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def group_auc(labels: np.ndarray, scores: np.ndarray, group_ids: np.ndarray) -> dict[str, float]:
    """Compute impression-weighted AUC per group and overall coverage."""
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for label, score, group_id in zip(labels, scores, group_ids, strict=True):
        grouped[int(group_id)].append((float(label), float(score)))
    covered_groups = 0
    weighted_auc_sum = 0.0
    weighted_count = 0
    for rows in grouped.values():
        group_labels = np.asarray([r[0] for r in rows], dtype=np.float64)
        group_scores = np.asarray([r[1] for r in rows], dtype=np.float64)
        if np.sum(group_labels > 0.5) == 0 or np.sum(group_labels <= 0.5) == 0:
            continue
        covered_groups += 1
        weighted_auc_sum += binary_auc(group_labels, group_scores) * len(rows)
        weighted_count += len(rows)
    coverage = covered_groups / max(len(grouped), 1)
    value = weighted_auc_sum / weighted_count if weighted_count else 0.5
    return {"value": float(value), "coverage": float(coverage)}


def compute_classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float | dict[str, float]]:
    """Compute AUC, log-loss, and group-AUC from raw logits."""
    probabilities = sigmoid(logits)
    return {
        "auc": binary_auc(labels, logits),
        "logloss": binary_logloss(labels, probabilities),
        "gauc": group_auc(labels, logits, group_ids),
    }
