import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_no_eval_debug_used() -> None:
    decision = json.loads((BASE / "sw14c_residual_learning_audit_decision.json").read_text(encoding="utf-8"))
    gamma = json.loads((BASE / "sw14c_gamma_sensitivity_summary.json").read_text(encoding="utf-8"))
    teacher = json.loads((BASE / "sw14c_teacher_error_region_summary.json").read_text(encoding="utf-8"))
    assert decision["whether_run_eval_debug_now"] is False
    assert decision["uses_eval_debug"] is False
    assert decision["uses_eval_core100_or_core500"] is False
    assert gamma["uses_eval_debug"] is False
    assert gamma["uses_eval_core100_or_core500"] is False
    assert teacher["uses_eval_debug"] is False
