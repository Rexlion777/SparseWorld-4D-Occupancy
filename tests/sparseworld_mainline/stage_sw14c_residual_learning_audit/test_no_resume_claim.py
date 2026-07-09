import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_no_resume_claim() -> None:
    decision = json.loads((BASE / "sw14c_residual_learning_audit_decision.json").read_text(encoding="utf-8"))
    report_text = (BASE / "stage_sw14c_residual_learning_audit_report.md").read_text(encoding="utf-8").lower()
    assert decision["whether_resume_allowed"] is False
    assert "official benchmark" not in report_text
    assert "sota" not in report_text
    assert "resume allowed" not in report_text
