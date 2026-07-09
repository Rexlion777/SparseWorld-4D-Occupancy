from __future__ import annotations

from conftest import load_json


def test_no_resume_claim_for_oracle_only() -> None:
    final = load_json("sw14c_oracle_error_mask_residual_final_decision.json")
    assert final["whether_use_in_resume"] is False
    assert final["sw13_remains_main_result"] is True
    assert final["whether_eval_debug_allowed"] is False
