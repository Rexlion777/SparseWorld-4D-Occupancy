from __future__ import annotations

from conftest import load_json


def test_no_eval_debug_used_unless_allowed() -> None:
    final = load_json("sw14e_three_route_final_decision.json")
    route1 = load_json("sw14d_b_final_decision.json")
    assert final["uses_eval_debug"] is False
    assert final["uses_core100_or_core500"] is False
    assert route1["uses_eval_debug"] is False
    assert route1["uses_core100_or_core500"] is False
    if final["whether_eval_debug_allowed"]:
        assert final["decision"] in {
            "SW14E_DECISION_1_SW14D_B_READY_DEBUG",
            "SW14E_DECISION_2_SW14D_B_SMALL_READY_DEBUG",
        }
