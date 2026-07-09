from __future__ import annotations

from conftest import load_json


def test_final_decision_schema() -> None:
    h = load_json("sw14h_final_decision.json")
    i = load_json("sw14i_final_decision.json")
    final = load_json("sw14hi_nuclear_add_rescue_final_decision.json")
    assert h["decision"] in {
        "SW14H_0_TOKEN_OR_DATA_FAIL",
        "SW14H_1_TRANSFORMER_NO_PRECISION_GAIN",
        "SW14H_2_PRECISION_GAIN_BUT_SELECTION_UNSAFE",
        "SW14H_3_SAFE_SUPPRESS_ONLY_NO_ADD_GAIN",
        "SW14H_4_SAFE_ADD_RECALL_NONREGRESSION_READY_DEBUG",
        "SW14H_5_SAFE_ADD_RECALL_GAIN_READY_DEBUG",
        "SW14H_6_PROTOCOL_BUG",
    }
    assert i["decision"] in {
        "SW14I_0_TARGET_OR_ROI_FAIL",
        "SW14I_1_AUTOENCODER_NO_SIGNAL",
        "SW14I_2_GENERATIVE_TOPK_PRECISION_GAIN_BUT_UNSAFE",
        "SW14I_3_GENERATIVE_PROPOSAL_SAFE_SMALL",
        "SW14I_4_GENERATIVE_PROPOSAL_STRONG_READY_DISTILL",
        "SW14I_5_DIFFUSION_DIAGNOSTIC_PROMISING",
        "SW14I_6_DIFFUSION_TOO_HEAVY_STOP",
        "SW14I_7_PROTOCOL_BUG",
    }
    assert final["decision"] in {
        "SW14HI_DECISION_1_TRANSFORMER_READY_DEBUG",
        "SW14HI_DECISION_2_TRANSFORMER_SMALL_READY_DEBUG",
        "SW14HI_DECISION_3_GENERATIVE_PROMISING_DISTILL",
        "SW14HI_DECISION_4_PRECISION_SIGNAL_UNSAFE",
        "SW14HI_DECISION_5_GENERATIVE_ORACLE_ONLY",
        "SW14HI_DECISION_6_ADD_BRANCH_HARD_STOP",
        "SW14HI_DECISION_7_PROTOCOL_INVALID",
    }
    assert "recommended_next_action" in final
    assert final["whether_sw13_remains_main_result"] is True
