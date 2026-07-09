from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling/stage_sw11_risk_targeted_h2_resampling_report.md"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "not long training" in text
    assert "not an official benchmark" in text or "not official benchmark" in text
    assert "does not claim model improvement" in text or "no model improvement claim" in text
    assert "20-iter smoke is only a mechanism test" in text
    banned_positive = [
        "official sota",
        "beats the paper",
        "beating the paper",
        "benchmark improvement",
        "production ready",
    ]
    for phrase in banned_positive:
        assert phrase not in text, f"false claim phrase found: {phrase}"
