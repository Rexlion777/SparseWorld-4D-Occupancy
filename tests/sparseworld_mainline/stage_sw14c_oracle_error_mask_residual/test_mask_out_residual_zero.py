from __future__ import annotations

from conftest import load_json


def test_mask_out_residual_zero_declared_and_enforced_structurally() -> None:
    init = load_json("sw14c_dual_branch_adapter_init_audit.json")
    health = load_json("sw14c_oracle_mask_residual_health_summary.json")
    assert init["voxel_to_feature_mask_alignment_available"] is False
    assert "output-space" in init["mask_out_enforcement"]
    assert health["feature_mask_alignment_available"] is False
    assert "output-space" in health["mask_out_residual_zero_method"]
    assert float(health["rear_residual"]) == 0.0
    assert float(health["clean_residual"]) == 0.0
