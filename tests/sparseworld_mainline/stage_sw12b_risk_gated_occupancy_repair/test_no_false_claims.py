from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/stage_sw12b_risk_gated_occupancy_repair_report.md"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "subset diagnostic" in text
    assert "not official benchmark" in text or "no official benchmark" in text
    assert "teacher prediction is not gt" in text
    assert "no claim of surpassing prior paper" in text or "no claim of surpassing" in text
    banned_positive = [
        "official sota",
        "beats the paper",
        "beating the paper",
        "final model improvement",
        "production ready",
    ]
    for phrase in banned_positive:
        assert phrase not in text, f"false claim phrase found: {phrase}"
