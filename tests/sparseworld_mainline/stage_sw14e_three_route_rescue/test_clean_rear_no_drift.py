from __future__ import annotations

from conftest import load_json


def test_clean_rear_no_drift() -> None:
    final = load_json("sw14e_three_route_final_decision.json")
    route1 = load_json("sw14d_b_final_decision.json")
    assert final["clean_drift"] == 0
    assert final["rear_unintended_drift"] == 0
    assert route1["clean_drift"] == 0
    assert route1["rear_unintended_drift"] == 0
