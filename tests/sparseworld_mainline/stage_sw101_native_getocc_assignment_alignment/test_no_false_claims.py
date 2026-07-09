from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment/stage_sw101_native_getocc_assignment_alignment_report.md"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "subset diagnostic only" in text
    assert "not official benchmark" in text
    assert "not training" in text
    banned_positive = [
        "official sota",
        "beats the paper",
        "beating the paper",
        "production ready",
        "calibrated uncertainty achieved",
        "full validation benchmark",
    ]
    for phrase in banned_positive:
        assert phrase not in text, f"false claim phrase found: {phrase}"
    assert "no model improvement claim" in text
