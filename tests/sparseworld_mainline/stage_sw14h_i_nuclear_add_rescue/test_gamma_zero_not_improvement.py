from __future__ import annotations

from conftest import load_json


def test_gamma_zero_not_improvement() -> None:
    for name in [
        "sw14h_final_decision.json",
        "sw14i_final_decision.json",
        "sw14hi_nuclear_add_rescue_final_decision.json",
    ]:
        assert load_json(name)["gamma_zero_used_as_improvement"] is False
