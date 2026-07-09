from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/keyframe_reconstruction_check_D_unified_nearest_match.json"


def test_horizon_time_mapping() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = {int(r["horizon_index"]): r for r in data["rows"]}
    assert rows[6]["model_time_sec"] == 3.0
