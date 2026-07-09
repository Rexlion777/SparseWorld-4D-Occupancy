from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/stage_swvis4_query_support_interpolation_report.md"
REPORT_JSON = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/stage_swvis4_query_support_interpolation_report.json"
FIX_REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/swvis4_endpoint_exact_fix_report.md"
FIX_REPORT_JSON = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/swvis4_endpoint_exact_fix_report.json"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    data = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert "not a 9s prediction" in text
    assert data["goal"] == "Fix keyframe jump in SW-VIS4 query/support interpolation with unified support-to-occupancy voxelization."
    assert "performance improvement" not in text


def test_endpoint_exact_fix_no_false_claims() -> None:
    text = FIX_REPORT.read_text(encoding="utf-8").lower()
    data = json.loads(FIX_REPORT_JSON.read_text(encoding="utf-8"))
    assert "not a 9s prediction" in text
    assert "not a model performance improvement" in text
    assert data["answers"]["is_model_performance_improvement"] is False
    assert data["answers"]["endpoint_residual_is_visualization_fix"] is True
