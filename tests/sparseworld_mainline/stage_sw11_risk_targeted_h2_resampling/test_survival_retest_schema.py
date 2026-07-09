from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling/sw11_alignment_survival_retest.csv"


def test_survival_retest_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty alignment survival retest csv"
    row = rows[0]
    required = [
        "checkpoint_name",
        "perturbation_id",
        "horizon_s",
        "scenario_name",
        "has_h2_high_score_ratio",
        "H2_native_IoU",
        "has_native_exact_assignment_ratio",
        "has_final_contributor_ratio",
        "GT_class_top3_survival_ratio",
    ]
    for key in required:
        assert key in row and row[key] != "", f"missing {key}"
