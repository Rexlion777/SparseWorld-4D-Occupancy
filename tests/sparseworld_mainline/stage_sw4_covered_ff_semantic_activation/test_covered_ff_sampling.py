from __future__ import annotations

import csv
from pathlib import Path


REPORTS = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation")
ARTIFACTS = Path("/mnt/d/ComputerVision/cv_lidar_transition/artifacts/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation")


def test_covered_ff_sampling_outputs_exist() -> None:
    csv_path = REPORTS / "covered_ff_voxel_sampling_summary.csv"
    pt_path = ARTIFACTS / "covered_ff_voxel_samples.pt"
    assert csv_path.exists()
    assert pt_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) > 0
    assert any(row["region_name"] == "covered_false_free" for row in rows)
