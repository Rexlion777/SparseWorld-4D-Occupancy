from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"


def test_query_coverage_outputs() -> None:
    rows = list(csv.DictReader((REPORTS / "query_coverage_metrics_sample_subset.csv").open(encoding="utf-8")))
    assert rows, "query coverage rows empty"
    radii = {int(row["coverage_radius_cells"]) for row in rows}
    assert radii == {1, 2, 3}
    first = rows[0]
    for col in [
        "query_coverage_gt_ratio",
        "query_coverage_false_free_ratio",
        "query_in_range_ratio",
        "query_density_entropy",
    ]:
        assert col in first
