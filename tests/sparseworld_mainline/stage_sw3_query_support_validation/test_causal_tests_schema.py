from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation"


def test_causal_tests_have_expected_interventions() -> None:
    payload = json.loads((REPORTS / "query_causal_controllability_tests.json").read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert rows, "causal test rows are empty"
    names = {row["intervention_name"] for row in rows}
    assert "refine_pts_current_shuffle_query" in names
    assert "refine_pts_current_center_replace" in names
    assert "cls_score_current_suppress_foreground" in names
    assert "forecast_points_h6_shuffle_query" in names
    assert "forecast_semantics_h6_suppress" in names


def test_causal_tests_cover_current_and_future_horizons() -> None:
    payload = json.loads((REPORTS / "query_causal_controllability_tests.json").read_text(encoding="utf-8"))
    horizons = {int(row["horizon_s"]) for row in payload["rows"]}
    assert 0 in horizons
    assert 6 in horizons
