from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/layered_interpolation_summary.json"


def test_no_raw_hard_cut() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    assert data["keyframe_source_mode"] == "endpoint_residual_blend"
    assert data["native_endpoint_anchor_mode"] == "native_get_occ_replay_exact"
    assert data["no_raw_hard_cut"] is True
