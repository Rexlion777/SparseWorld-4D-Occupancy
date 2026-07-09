from __future__ import annotations

from conftest import load_json


def test_f3_frontcap_unchanged() -> None:
    inherited = load_json("sw14c_oracle_mask_inherited_state.json")
    final = load_json("sw14c_oracle_error_mask_residual_final_decision.json")
    assert inherited["f3_script_sha256_before"] == final["f3_script_sha256_after"]
    assert inherited["frontcap_script_sha256_before"] == final["frontcap_script_sha256_after"]
