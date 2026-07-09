from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/keyframe_reconstruction_check_D_unified_nearest_match.json"


def test_keyframe_reconstruction_exists_and_schema() -> None:
    assert REPORT.exists()
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    assert data["variant"] == "D_unified_nearest_match"
    assert len(data["rows"]) == 7
    assert 0.0 <= data["mean_occupied_iou"] <= 1.0
    assert 0.0 <= data["mean_semantic_agreement_intersection"] <= 1.0
    assert data["mean_occupied_count_ratio"] > 0.0
