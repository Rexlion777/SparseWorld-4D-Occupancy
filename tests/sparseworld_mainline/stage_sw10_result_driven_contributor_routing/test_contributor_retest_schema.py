from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing/sw10_contributor_diagnostic_retest.csv"


def test_contributor_retest_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty contributor retest csv"
    row = rows[0]
    assert "checkpoint_name" in row
    assert "radius_support_coverage" in row or "skipped_reason" in row
