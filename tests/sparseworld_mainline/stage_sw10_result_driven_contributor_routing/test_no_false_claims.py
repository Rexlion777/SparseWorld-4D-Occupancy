from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing/stage_sw10_result_driven_contributor_routing_report.md"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "not official benchmark" in text
    banned_positive = [
        "beats the paper",
        "beating the paper",
        "official sota",
        "production ready",
        "calibrated uncertainty achieved",
    ]
    for phrase in banned_positive:
        assert phrase not in text, f"false claim phrase found: {phrase}"
    assert "false-positive / pred_gt_ratio" in text or "false_occupied_delta" in text
