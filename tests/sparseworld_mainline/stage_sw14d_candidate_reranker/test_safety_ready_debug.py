from __future__ import annotations

from conftest import load_json


def test_safety_ready_debug() -> None:
    summary = load_json("sw14d_candidate_reranker_eval_summary.json")["val"]
    final = load_json("sw14d_candidate_reranker_final_decision.json")
    assert summary["safety_pass_all"] is True
    if final["whether_eval_debug_allowed"]:
        assert float(summary["mean_net_score"]) >= 0.002
        assert summary["front_fn_non_degraded"] is True
        assert float(summary["mean_front_fn_reduction_rate"]) >= 0.0
    else:
        assert final["decision"] in {
            "SW14D_R2A_SUPPRESS_SAFE_FRONT_FN_DEGRADED",
            "SW14D_R3_SIGNAL_ONLY_UNSAFE",
            "SW14D_R4_WEAK_SIGNAL_NO_DEBUG",
            "SW14D_R5_NO_SIGNAL_KEEP_SW13",
        }
