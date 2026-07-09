import json
from pathlib import Path


PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter/sw14c_final_decision.json")


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision"] in {
        "SW14C_0_FAIL_KEEP_SW13_MAIN",
        "SW14C_1_DEBUG_IMPROVES_TEACHER",
        "SW14C_2_CORE100_IMPROVES_TEACHER",
        "SW14C_3_CORE500_IMPROVES_TEACHER",
        "SW14C_4_COMPARABLE_BUT_NOT_STRONGER",
        "SW14C_5_UNSAFE_DENSITY",
        "SW14C_6_PROTOCOL_VIOLATION",
    }
    required = {
        "best_checkpoint",
        "best_gamma",
        "whether_beats_sw13_teacher",
        "whether_use_in_resume",
        "whether_SW13_remains_main",
        "safe_claim",
        "limitations",
        "next_action",
    }
    assert required.issubset(payload.keys())
