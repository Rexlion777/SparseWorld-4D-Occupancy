from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT_MD = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/stage_sw7_sensor_conditioned_reliability_map_report.md"
REPORT_JSON = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/stage_sw7_sensor_conditioned_reliability_map_report.json"


def test_no_false_claims() -> None:
    md = REPORT_MD.read_text(encoding="utf-8").lower()
    js = REPORT_JSON.read_text(encoding="utf-8").lower()
    text = md + "\n" + js
    assert "not official benchmark" in text
    assert "not calibrated uncertainty" in text
    assert "not production-ready planning policy" in text or "not connected to production planning" in text
    forbidden = [
        "official benchmark improvement",
        "calibrated uncertainty estimate",
        "production-ready planning deployment",
        "model performance improvement claim: yes",
        "training completed",
    ]
    for token in forbidden:
        assert token not in text
