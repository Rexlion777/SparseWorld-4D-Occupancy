from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation/raw_interpolation_tensor_audit.json"


def test_raw_tensor_audit_exists() -> None:
    assert REPORT.exists()
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    assert data["has_horizon_wise_forecast_points"] is True
    assert data["has_horizon_wise_forecast_semantics"] is True
