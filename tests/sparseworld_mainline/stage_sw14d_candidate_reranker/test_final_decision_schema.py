from __future__ import annotations

from conftest import load_json


def test_final_decision_schema() -> None:
    final = load_json("sw14d_candidate_reranker_final_decision.json")
    allowed = {
        "SW14D_R1_SAFE_STRONG_GAIN_READY_DEBUG",
        "SW14D_R2_SAFE_SMALL_GAIN_READY_DEBUG",
        "SW14D_R2A_SUPPRESS_SAFE_FRONT_FN_DEGRADED",
        "SW14D_R3_SIGNAL_ONLY_UNSAFE",
        "SW14D_R4_WEAK_SIGNAL_NO_DEBUG",
        "SW14D_R5_NO_SIGNAL_KEEP_SW13",
    }
    assert final["decision"] in allowed
    assert "recommended_next_action" in final
    assert final["sw13_remains_main_result"] is True
    assert final["whether_use_in_resume"] is False
