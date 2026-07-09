from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
CSV_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/keyframe_exactness_check.csv"


def test_keyframe_exactness_schema() -> None:
    rows = list(csv.DictReader(CSV_PATH.open("r", encoding="utf-8")))
    assert rows
    required = {
        "variant",
        "horizon_index",
        "frame_index",
        "model_time_sec",
        "occupied_iou",
        "semantic_agreement_intersection",
        "occupied_count_ratio",
    }
    assert required.issubset(rows[0].keys())
    layered = [r for r in rows if r["variant"] == "endpoint_exact_layered"]
    assert len(layered) == 7
