from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment/h2_native_assignment_alignment_metrics.csv"


def test_alignment_metrics_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty alignment metrics csv"
    row = rows[0]
    required = [
        "checkpoint_name",
        "perturbation_id",
        "sample_index",
        "horizon_s",
        "h2_score_threshold",
        "sector_name",
        "class_group",
        "error_type",
        "H2_to_native_precision",
        "H2_to_native_recall",
        "H2_native_IoU",
    ]
    for key in required:
        assert key in row and row[key] != "", f"missing {key}"
