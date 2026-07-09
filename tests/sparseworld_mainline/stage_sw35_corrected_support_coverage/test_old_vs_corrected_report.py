from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage"


def test_old_vs_corrected_report_marks_old_conclusion_deprecated() -> None:
    text = (REPORTS / "old_vs_corrected_support_proxy_comparison.md").read_text(encoding="utf-8").lower()
    assert "deprecated" in text
    assert "mean-center" in text
    assert "all48" in text


def test_final_report_deprecates_old_coverage_failure_conclusion() -> None:
    payload = json.loads((REPORTS / "stage_sw35_corrected_support_coverage_report.json").read_text(encoding="utf-8"))
    assert payload["executive_summary"]["old_conclusion_deprecated"] is True
    assert payload["executive_summary"]["corrected_resolution"] == "covered_false_free_semantic_activation_dominant"
