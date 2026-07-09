from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/sw7_reliability_config.json"


def test_reliability_config_schema() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert set(payload["weights"]) == {"semantic", "support", "contributor", "sensor", "temporal"}
    assert 0.0 <= payload["thresholds"]["high_risk"] <= 1.0
    assert isinstance(payload["thresholds"]["risk_curve_thresholds"], list)
    assert payload["safe_claim"] == "internal diagnostic reliability indicator, not calibrated uncertainty"
