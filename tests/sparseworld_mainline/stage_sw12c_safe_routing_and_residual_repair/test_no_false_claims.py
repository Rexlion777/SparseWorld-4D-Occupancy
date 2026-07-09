from pathlib import Path

REPORT = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/stage_sw12c_safe_routing_and_residual_repair_report.md")


def test_no_false_claims() -> None:
    text = REPORT.read_text(encoding="utf-8").lower()
    assert "subset diagnostic" in text
    assert "not official benchmark" in text
    banned = ["official benchmark result", "final model improvement", "beats the paper", "production ready"]
    for phrase in banned:
        assert phrase not in text
