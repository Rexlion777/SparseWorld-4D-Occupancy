from __future__ import annotations

from conftest import load_json


def test_gamma_zero_not_improvement() -> None:
    final = load_json("sw14e_three_route_final_decision.json")
    route1 = load_json("sw14d_b_final_decision.json")
    assert final["gamma_zero_not_used_as_improvement"] is True
    assert route1["gamma_zero_not_used_as_improvement"] is True
