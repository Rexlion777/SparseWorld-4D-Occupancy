import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")
ENUMS = {
    "SW14C_R2B_0_STILL_NOOP",
    "SW14C_R2B_1_SIGNAL_ONLY_UNSAFE",
    "SW14C_R2B_2_SAFE_BUT_NO_MEANINGFUL_GAIN",
    "SW14C_R2B_3_SAFE_SMALL_GAIN_READY_DEBUG",
    "SW14C_R2B_4_SAFE_STRONG_GAIN_READY_DEBUG",
    "SW14C_R2B_5_BREAKS_TEACHER_CORRECT",
    "SW14C_R2B_6_MASK_OR_PROTOCOL_BUG",
    "SW14C_R2B_7_STOP_KEEP_SW13",
}


def test_decision_schema():
    obj = json.loads((BASE / "sw14c_round2b_final_decision.json").read_text())
    required = {
        "best_checkpoint",
        "selected_gamma",
        "selected_gamma_is_nonzero",
        "val_over_teacher_improvement",
        "density_safety",
        "fp_safety",
        "front_local_safety",
        "clean_rear_safety",
        "residual_health_decision",
        "teacher_error_behavior_decision",
        "whether_eval_debug_allowed",
        "whether_round3_allowed",
        "whether_resume_allowed",
        "recommended_next_action",
    }
    assert obj["decision"] in ENUMS
    assert required.issubset(obj)
