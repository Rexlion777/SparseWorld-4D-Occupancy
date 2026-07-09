from __future__ import annotations

from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage"


def test_sw35_no_false_claims() -> None:
    text = (REPORTS / "stage_sw35_corrected_support_coverage_report.md").read_text(encoding="utf-8").lower()
    disallowed_positive = [
        "official benchmark reproduced",
        "training completed.",
        "model improved",
        "sensor perturbation completed.",
    ]
    for phrase in disallowed_positive:
        assert phrase not in text, f"disallowed positive claim found: {phrase}"
    assert "deprecated" in text
    assert "corrected support definition" in text
