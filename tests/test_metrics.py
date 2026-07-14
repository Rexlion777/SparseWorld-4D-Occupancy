import pytest

import numpy as np

from sparseworld_reliability import occupancy_metrics, relative_error_reduction


def test_relative_error_reduction() -> None:
    assert relative_error_reduction(1.0, 0.768) == pytest.approx(0.232)


def test_relative_error_reduction_rejects_zero_baseline() -> None:
    with pytest.raises(ValueError):
        relative_error_reduction(0.0, 0.0)


def test_occupancy_metrics_report_recovery_and_risk_together() -> None:
    target = np.array([1, 1, 17, 17])
    prediction = np.array([1, 17, 2, 17])
    metrics = occupancy_metrics(prediction, target)
    assert metrics["occupied_iou"] == pytest.approx(1 / 3)
    assert metrics["false_negative"] == 1
    assert metrics["false_occupied"] == 1
    assert metrics["predicted_to_target_density"] == pytest.approx(1.0)
