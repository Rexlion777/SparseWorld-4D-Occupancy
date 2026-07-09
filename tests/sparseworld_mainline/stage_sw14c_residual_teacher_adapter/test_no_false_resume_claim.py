import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter")


def test_no_false_resume_claim() -> None:
    decision = json.loads((BASE / "sw14c_final_decision.json").read_text(encoding="utf-8"))
    report_text = (BASE / "stage_sw14c_residual_teacher_adapter_report.md").read_text(encoding="utf-8").lower()
    if decision["decision"] not in {"SW14C_1_DEBUG_IMPROVES_TEACHER", "SW14C_2_CORE100_IMPROVES_TEACHER", "SW14C_3_CORE500_IMPROVES_TEACHER"}:
        assert decision["whether_use_in_resume"] is False
        assert decision["whether_beats_sw13_teacher"] is False
    assert "official benchmark result" not in report_text
    assert "sota" not in report_text
