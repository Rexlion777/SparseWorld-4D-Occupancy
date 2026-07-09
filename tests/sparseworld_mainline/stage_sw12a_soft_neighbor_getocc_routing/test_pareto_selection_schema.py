from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing/sw12a_pareto_candidates.csv"


def test_pareto_selection_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected non-empty pareto candidate table"
    allowed = {
        "PARETO_SAFE",
        "TARGETED_BUT_UNSAFE",
        "SAFE_BUT_NO_TARGET",
        "ORACLE_ONLY_SIGNAL",
        "NO_SIGNAL",
    }
    for row in rows:
        assert row["classification"] in allowed
        assert row["variant_label"]
