import json
from pathlib import Path


PATH = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter/sw14c_gamma_selection.json")


def test_gamma_selection_val_only() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["uses_eval_debug_for_selection"] is False
    assert payload["gamma_zero_counts_as_improved_result"] is False
    assert payload.get("uses_val_small_for_selection") is True or payload.get("uses_cache_diagnostic_for_selection") is True
    if payload.get("uses_cache_diagnostic_for_selection") is True:
        assert payload.get("protocol_final_claim_allowed") is False
    assert payload["decision"] in {
        "GAMMA_R1_SAFE_IMPROVES_TEACHER",
        "GAMMA_R2_SAFE_NO_IMPROVEMENT",
        "GAMMA_R3_UNSAFE",
        "GAMMA_R4_PROTOCOL_BUG",
    }
