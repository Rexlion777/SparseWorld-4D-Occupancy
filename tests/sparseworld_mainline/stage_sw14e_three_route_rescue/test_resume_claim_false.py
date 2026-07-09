from __future__ import annotations

from conftest import load_json


def test_resume_claim_false() -> None:
    final = load_json("sw14e_three_route_final_decision.json")
    route1 = load_json("sw14d_b_final_decision.json")
    route2 = load_json("sw14f_final_decision.json")
    route3 = load_json("sw14g_final_decision.json")
    assert final["whether_resume_claim_allowed"] is False
    assert route1["whether_resume_claim_allowed"] is False
    assert route2["whether_resume_claim_allowed"] is False
    assert route3["whether_resume_claim_allowed"] is False
