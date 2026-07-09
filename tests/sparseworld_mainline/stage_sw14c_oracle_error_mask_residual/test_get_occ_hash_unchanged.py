from __future__ import annotations

from conftest import load_json


def test_get_occ_hash_unchanged() -> None:
    inherited = load_json("sw14c_oracle_mask_inherited_state.json")
    final = load_json("sw14c_oracle_error_mask_residual_final_decision.json")
    assert inherited["get_occ_sha256_before"] == final["get_occ_sha256_after"]
