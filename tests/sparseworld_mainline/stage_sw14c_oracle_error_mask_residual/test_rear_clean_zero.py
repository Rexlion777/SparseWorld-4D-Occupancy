from __future__ import annotations

from conftest import load_json


def test_rear_clean_zero() -> None:
    init = load_json("sw14c_dual_branch_adapter_init_audit.json")
    health = load_json("sw14c_oracle_mask_residual_health_summary.json")
    assert float(init["rear_residual_max_abs"]) == 0.0
    assert float(init["clean_residual_max_abs"]) == 0.0
    assert float(health["rear_residual"]) == 0.0
    assert float(health["clean_residual"]) == 0.0
