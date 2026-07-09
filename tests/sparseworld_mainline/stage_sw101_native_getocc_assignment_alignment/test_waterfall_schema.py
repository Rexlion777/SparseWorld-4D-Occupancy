from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment/h2_to_semantic_occ_waterfall.csv"


def test_waterfall_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty waterfall csv"
    row = rows[0]
    required = [
        "checkpoint_name",
        "perturbation_id",
        "sample_index",
        "horizon_s",
        "scenario_name",
        "voxel_count",
        "has_h2_high_score_ratio",
        "has_native_exact_assignment_ratio",
        "has_final_contributor_ratio",
        "final_semantic_occ_correct_ratio",
    ]
    for key in required:
        assert key in row and row[key] != "", f"missing {key}"
