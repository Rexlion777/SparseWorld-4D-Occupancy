from __future__ import annotations

from conftest import load_json


def test_final_decision_schema() -> None:
    final = load_json("sw14e_three_route_final_decision.json")
    allowed = {
        "SW14E_DECISION_1_SW14D_B_READY_DEBUG",
        "SW14E_DECISION_2_SW14D_B_SMALL_READY_DEBUG",
        "SW14E_DECISION_3_FEATURE_POSTPROCESS_AWARE_HAS_SIGNAL",
        "SW14E_DECISION_4_QUERY_LEVEL_ADAPTER_PROMISING",
        "SW14E_DECISION_5_KEEP_SW14D_A_DIAGNOSTIC_ONLY",
        "SW14E_DECISION_6_STOP_SW14_KEEP_SW13_MAIN",
        "SW14E_DECISION_7_PROTOCOL_INVALID",
    }
    assert final["decision"] in allowed
    assert final["sw13_remains_main_result"] is True
    assert final["whether_round3_allowed"] is False
    assert final["whether_resume_claim_allowed"] is False
    assert "recommended_next_unique_action" in final
