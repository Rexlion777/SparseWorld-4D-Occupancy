from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing/stage_sw12a_soft_neighbor_getocc_routing_report.md"


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "diagnostic replay only" in text
    assert "native get_occ default is unchanged".lower() in text or "native get_occ default unchanged" in text
    assert "no long training" in text
    assert "no official benchmark" in text or "not official benchmark" in text
    assert "no model improvement claim" in text
    assert "oracle is diagnostic upper bound only" in text or "v4 oracle is diagnostic upper bound only" in text
    banned_positive = [
        "official sota",
        "beats the paper",
        "beating the paper",
        "model improvement",
        "production ready",
    ]
    for phrase in banned_positive:
        if phrase == "model improvement":
            continue
        assert phrase not in text, f"false claim phrase found: {phrase}"
