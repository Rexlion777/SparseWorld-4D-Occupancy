from __future__ import annotations

from conftest import load_json


def test_no_eval_debug_used() -> None:
    inherited = load_json("sw14c_oracle_mask_inherited_state.json")
    final = load_json("sw14c_oracle_error_mask_residual_final_decision.json")
    gamma = load_json("sw14c_oracle_mask_gamma_selection.json")
    assert inherited["uses_eval_debug"] is False
    assert final["uses_eval_debug"] is False
    assert final["whether_eval_debug_allowed"] is False
    assert gamma["uses_eval_debug_for_selection"] is False
