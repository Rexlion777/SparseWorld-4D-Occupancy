from __future__ import annotations

from conftest import load_json


def test_decision_schema() -> None:
    final = load_json("sw14c_oracle_error_mask_residual_final_decision.json")
    allowed = {
        "SW14C_OEM_0_ORACLE_MASK_FAIL_KEEP_SW13",
        "SW14C_OEM_1_ORACLE_MASK_LOW_POTENTIAL",
        "SW14C_OEM_2_ORACLE_MASK_SIGNAL_UNSAFE",
        "SW14C_OEM_3_ORACLE_MASK_SAFE_GAIN",
        "SW14C_OEM_4_PROXY_MASK_SAFE_GAIN_READY_DEBUG",
        "SW14C_OEM_5_PROXY_FAIL_BUT_ORACLE_SUCCESS",
        "SW14C_OEM_6_DUAL_BRANCH_BREAKS_CORRECT",
        "SW14C_OEM_7_PROTOCOL_VIOLATION",
    }
    required = [
        "decision",
        "oracle_has_potential",
        "masked_residual_noop_fixed",
        "fn_fp_behavior_targeted",
        "proxy_mask_feasible",
        "whether_eval_debug_allowed",
        "whether_learned_error_mask_next",
        "whether_use_in_resume",
        "sw13_remains_main_result",
        "best_checkpoint",
        "selected_gamma_add",
        "selected_gamma_sup",
        "val_net_improvement",
        "recommended_next_action",
    ]
    assert final["decision"] in allowed
    for key in required:
        assert key in final
    assert final["sw13_remains_main_result"] is True
