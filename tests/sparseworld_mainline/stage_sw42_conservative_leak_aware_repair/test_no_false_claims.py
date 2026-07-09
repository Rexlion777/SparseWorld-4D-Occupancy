from pathlib import Path

BASE = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair")


def test_no_false_claims():
    text = (BASE / "stage_sw42_conservative_leak_aware_repair_report.md").read_text(encoding="utf-8").lower()
    forbidden = [
        "official benchmark reproduced",
        "training completed",
    ]
    assert not any(tok in text for tok in forbidden)
    assert "accepted repair" not in text or "k7 aggressive is rejected" in text
