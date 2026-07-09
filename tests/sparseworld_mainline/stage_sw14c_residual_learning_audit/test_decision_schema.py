import json
from pathlib import Path


PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit/sw14c_residual_learning_audit_decision.json")


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["primary_cause"] in {
        "SW14C_AUDIT_1_RESIDUAL_ZERO_COLLAPSE",
        "SW14C_AUDIT_2_GATE_OR_REG_TOO_STRONG",
        "SW14C_AUDIT_3_TEACHER_ERROR_TOO_SPARSE",
        "SW14C_AUDIT_4_RAW_CHANGED_BUT_POSTPROCESS_SWALLOWED",
        "SW14C_AUDIT_5_WRONG_REGION_BEHAVIOR",
        "SW14C_AUDIT_6_SIGNAL_ONLY_UNSAFE",
        "SW14C_AUDIT_7_READY_FOR_ROUND3_WITH_LOSS_ADJUST",
        "SW14C_AUDIT_8_STOP_SW14C_KEEP_SW13",
    }
    required = {
        "primary_cause",
        "secondary_causes",
        "evidence",
        "recommended_next_action",
        "whether_run_eval_debug_now",
        "whether_round3_allowed",
        "whether_teacher_error_mining_needed",
        "whether_loss_adjustment_needed",
        "whether_resume_allowed",
    }
    assert required.issubset(payload.keys())
    assert payload["final_claim_allowed"] is False
    assert payload["subset_diagnostic_only"] is True
