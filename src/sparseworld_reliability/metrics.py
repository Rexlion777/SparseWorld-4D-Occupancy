"""Small metric helpers with explicit failure-recovery semantics."""

from __future__ import annotations

import numpy as np


def relative_error_reduction(degraded_error: float, repaired_error: float) -> float:
    """Return relative reduction in an error metric, expressed as a fraction.

    Positive values indicate improvement. For example, 0.232 means a 23.2%
    relative reduction against the degraded baseline.
    """

    if degraded_error <= 0:
        raise ValueError("degraded_error must be positive")
    return (degraded_error - repaired_error) / degraded_error


def occupancy_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    empty_index: int = 17,
) -> dict[str, float | int]:
    """Compute task and safety metrics from semantic occupancy grids.

    The result deliberately reports false negatives and false occupancy next to
    occupied IoU, preventing a recall-only repair from looking artificially good.
    """

    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} != {target.shape}")

    predicted_occupied = prediction != empty_index
    target_occupied = target != empty_index
    true_positive = int(np.count_nonzero(predicted_occupied & target_occupied))
    false_negative = int(np.count_nonzero(~predicted_occupied & target_occupied))
    false_occupied = int(np.count_nonzero(predicted_occupied & ~target_occupied))
    union = int(np.count_nonzero(predicted_occupied | target_occupied))
    target_count = int(np.count_nonzero(target_occupied))

    return {
        "occupied_iou": true_positive / union if union else 1.0,
        "false_negative": false_negative,
        "false_occupied": false_occupied,
        "predicted_occupied": int(np.count_nonzero(predicted_occupied)),
        "target_occupied": target_count,
        "predicted_to_target_density": (
            float(np.count_nonzero(predicted_occupied)) / target_count if target_count else 0.0
        ),
    }
