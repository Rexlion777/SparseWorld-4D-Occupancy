from __future__ import annotations

import json
from pathlib import Path


REPORTS = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation")


def test_sw4_no_false_claims() -> None:
    report_json = REPORTS / "stage_sw4_covered_ff_semantic_activation_report.json"
    report_md = REPORTS / "stage_sw4_covered_ff_semantic_activation_report.md"
    decision_json = REPORTS / "sw4_covered_ff_failure_taxonomy_decision.json"
    assert report_json.exists()
    assert report_md.exists()
    assert decision_json.exists()

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["safe_claims"]["subset_diagnostic_only"] is True
    assert payload["safe_claims"]["official_benchmark"] is False
    assert payload["safe_claims"]["training_completed"] is False
    assert payload["safe_claims"]["oracle_interventions_are_diagnostic_only"] is True
    assert payload["safe_claims"]["model_improvement_claim"] is False

    text = report_md.read_text(encoding="utf-8").lower()
    banned = [
        "official benchmark reproduced",
        "training completed",
        "model improvement claim: true",
        "sota",
        "full benchmark",
    ]
    for item in banned:
        assert item not in text
