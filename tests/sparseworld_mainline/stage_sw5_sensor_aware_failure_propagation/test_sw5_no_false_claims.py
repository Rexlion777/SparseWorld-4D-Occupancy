from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation/stage_sw5_sensor_aware_failure_propagation_report.md"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    banned = [
        "official benchmark",
        "production-ready",
        "training completed",
    ]
    for token in banned:
        assert token not in text or "not " + token in text
