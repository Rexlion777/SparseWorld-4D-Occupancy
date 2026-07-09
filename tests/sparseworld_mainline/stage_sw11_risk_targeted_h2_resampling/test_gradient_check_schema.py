from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling/sw11_gradient_check.csv"


def test_gradient_check_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty gradient check csv"
    row = rows[0]
    required = [
        "experiment_id",
        "sampler_mode",
        "total_loss",
        "original_loss",
        "h2_assign_loss",
        "h2_leak_loss",
        "positive_target_count",
        "fallback_ratio",
        "grad_finite",
        "eligible_for_smoke",
    ]
    for key in required:
        assert key in row and row[key] != "", f"missing {key}"
