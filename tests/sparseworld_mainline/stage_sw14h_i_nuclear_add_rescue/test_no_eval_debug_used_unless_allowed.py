from __future__ import annotations

from conftest import load_json


def test_no_eval_debug_used_unless_allowed() -> None:
    final = load_json("sw14hi_nuclear_add_rescue_final_decision.json")
    h = load_json("sw14h_final_decision.json")
    i = load_json("sw14i_final_decision.json")
    assert final["uses_eval_debug"] is False
    assert h["uses_eval_debug"] is False
    assert i["uses_eval_debug"] is False
    if not final["whether_eval_debug_allowed"]:
        assert final["decision"] not in {
            "SW14HI_DECISION_1_TRANSFORMER_READY_DEBUG",
            "SW14HI_DECISION_2_TRANSFORMER_SMALL_READY_DEBUG",
        }


def test_no_core100_or_core500_used() -> None:
    for name in ["sw14hi_nuclear_add_rescue_final_decision.json", "sw14h_final_decision.json", "sw14i_final_decision.json"]:
        assert load_json(name)["uses_core100_or_core500"] is False
