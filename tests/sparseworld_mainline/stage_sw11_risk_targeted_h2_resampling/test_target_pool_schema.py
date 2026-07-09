from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling/sw11_failure_target_pool_manifest.csv"


def test_target_pool_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty target pool manifest"
    row = rows[0]
    required = [
        "pool_name",
        "checkpoint_source",
        "perturbation",
        "sample_id",
        "horizon",
        "voxel_count",
        "sector_distribution",
        "class_group_distribution",
        "error_type_distribution",
        "enough_samples",
    ]
    for key in required:
        assert key in row and row[key] != "", f"missing {key}"
