from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/branchB_teacher_cache_manifest.csv"


def test_teacher_cache_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    required = {"split", "sample_index", "horizon_s", "cache_path", "teacher_conf_thr_07_count", "teacher_occupied_count"}
    assert required <= set(rows[0].keys())
