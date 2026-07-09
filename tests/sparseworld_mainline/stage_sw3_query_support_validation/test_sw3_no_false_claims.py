from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation"


def test_no_false_claims_in_report() -> None:
    report_text = (REPORTS / "stage_sw3_query_support_validation_report.md").read_text(encoding="utf-8").lower()
    banned = [
        "official benchmark",
        "training completed",
        "sensor perturbation completed",
        "complete query lifecycle",
    ]
    for phrase in banned:
        assert phrase not in report_text, f"banned phrase found: {phrase}"


def test_no_unjustified_query_allocation_confirmation() -> None:
    decision = json.loads((REPORTS / "sw3_query_support_resolution_decision.json").read_text(encoding="utf-8"))
    report_text = (REPORTS / "stage_sw3_query_support_validation_report.md").read_text(encoding="utf-8").lower()
    if decision["resolution_case"] not in {"C", "D", "E"}:
        assert "query allocation confirmed" not in report_text
