from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RANKING = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/high_risk_region_ranking.json"
SCHEMA = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/planning_facing_risk_interface_schema.json"
DEMO = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/planning_facing_risk_interface_demo.json"


def test_high_risk_schema() -> None:
    ranking = json.loads(RANKING.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    demo = json.loads(DEMO.read_text(encoding="utf-8"))
    assert ranking["rows"]
    first = ranking["rows"][0]["risk_regions"][0]
    assert {"region_id", "sector", "class_group", "risk_score", "dominant_reason", "recommended_downstream_action"}.issubset(first)
    assert "risk_regions" in schema
    assert {"sample_idx", "perturbation", "horizon", "risk_regions", "safe_claim"}.issubset(demo)
