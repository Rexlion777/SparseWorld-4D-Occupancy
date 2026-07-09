from __future__ import annotations

from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"


def test_no_false_claims() -> None:
    text = (REPORTS / "stage_sw2_temporal_query_diagnosis_report.md").read_text(encoding="utf-8").lower()
    banned = [
        "official benchmark",
        "full validation",
        "sota reproduced",
        "training completed",
        "complete query lifecycle",
        "sensor-aware perturbation completed",
    ]
    for phrase in banned:
        assert phrase not in text, f"banned phrase found: {phrase}"
    assert "subset diagnostic" in text
