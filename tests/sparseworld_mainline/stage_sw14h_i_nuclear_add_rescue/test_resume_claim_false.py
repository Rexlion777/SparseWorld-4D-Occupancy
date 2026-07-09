from __future__ import annotations

from conftest import load_json


def test_resume_claim_false() -> None:
    for name in [
        "sw14h_final_decision.json",
        "sw14i_final_decision.json",
        "sw14hi_nuclear_add_rescue_final_decision.json",
    ]:
        assert load_json(name)["whether_resume_claim_allowed"] is False
