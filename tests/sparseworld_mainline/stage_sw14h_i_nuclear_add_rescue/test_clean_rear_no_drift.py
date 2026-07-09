from __future__ import annotations

from conftest import load_json


def test_clean_rear_no_drift() -> None:
    for name in [
        "sw14h_final_decision.json",
        "sw14i_final_decision.json",
        "sw14hi_nuclear_add_rescue_final_decision.json",
    ]:
        payload = load_json(name)
        assert payload["clean_drift"] == 0
        assert payload["rear_unintended_drift"] == 0
